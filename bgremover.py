#!/usr/bin/env python3

import os
import sys
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional, Dict
import multiprocessing
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from rembg import remove, new_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global session cache for each process
_session_cache: Dict[int, 'onnxruntime.InferenceSession'] = {}

# Maximum dimensions for different processing modes
MAX_DIMENSIONS = {
    'fast': 800,
    'balanced': 1200,
    'quality': 1500,
    'alpha_matting': 1000  # Maximum size for alpha matting
}

def get_session():
    """Get or create an ONNX session for the current process."""
    process_id = os.getpid()
    if process_id not in _session_cache:
        _session_cache[process_id] = new_session()
    return _session_cache[process_id]

def calculate_target_size(image_size: int, max_size: int, mode: str) -> int:
    """Calculate optimal target size based on image size and processing mode."""
    if mode == 'fast':
        return min(MAX_DIMENSIONS['fast'], max_size)
    
    if image_size > 2000:
        return min(MAX_DIMENSIONS['balanced'], max_size)
    elif image_size > 1000:
        return min(MAX_DIMENSIONS['quality'], max_size)
    else:
        return min(image_size, max_size)

def process_single_image(
    image_path: Path,
    output_dir: Path,
    prefix: str,
    suffix: str,
    max_size: int,
    alpha_matting: bool,
    alpha_matting_foreground_threshold: int,
    alpha_matting_background_threshold: int,
    alpha_matting_erode_size: int,
    post_process_mask: bool,
    fast_mode: bool = False
) -> Tuple[bool, str]:
    """Process a single image and return success status and message."""
    try:
        # Generate output filename
        output_filename = f"{prefix}{image_path.stem}{suffix}.png"
        output_path = output_dir / output_filename
        
        # Skip if output file already exists
        if output_path.exists():
            return True, f"Skipped {image_path.name} (already processed)"
        
        # Get cached session
        session = get_session()
        
        # Read and process image
        with Image.open(image_path) as input_image:
            # Convert to RGB if necessary
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
            
            # Calculate optimal size
            mode = 'fast' if fast_mode else 'balanced'
            target_size = calculate_target_size(max(input_image.size), max_size, mode)
            
            # Resize if necessary
            if max(input_image.size) > target_size:
                ratio = target_size / max(input_image.size)
                new_size = tuple(int(dim * ratio) for dim in input_image.size)
                input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # First try without alpha matting for speed
            try:
                output_image = remove(
                    input_image,
                    session=session,
                    alpha_matting=False
                )
                
                # If alpha matting is enabled and not in fast mode, try to improve edges
                if alpha_matting and not fast_mode:
                    try:
                        # Use more aggressive alpha matting parameters
                        am_fg = 250
                        am_bg = 5
                        am_erode = 20
                        # Check if image needs alpha matting and is within size limits
                        if (_needs_alpha_matting(output_image) and 
                            max(input_image.size) <= MAX_DIMENSIONS['alpha_matting']):
                            
                            # Create a smaller copy for alpha matting if needed
                            if max(input_image.size) > MAX_DIMENSIONS['alpha_matting']:
                                alpha_size = MAX_DIMENSIONS['alpha_matting']
                                ratio = alpha_size / max(input_image.size)
                                alpha_size = tuple(int(dim * ratio) for dim in input_image.size)
                                alpha_input = input_image.resize(alpha_size, Image.Resampling.LANCZOS)
                            else:
                                alpha_input = input_image
                            
                            # Apply alpha matting to the smaller image
                            alpha_output = remove(
                                alpha_input,
                                session=session,
                                alpha_matting=True,
                                alpha_matting_foreground_threshold=am_fg,
                                alpha_matting_background_threshold=am_bg,
                                alpha_matting_erode_size=am_erode
                            )
                            
                            # Resize back to original size if needed
                            if max(input_image.size) > MAX_DIMENSIONS['alpha_matting']:
                                alpha_output = alpha_output.resize(input_image.size, Image.Resampling.LANCZOS)
                            
                            output_image = alpha_output
                            
                    except Exception as e:
                        logger.warning(f"Alpha matting failed for {image_path.name}, using basic removal: {str(e)}")
                
            except Exception as e:
                logger.error(f"Background removal failed for {image_path.name}: {str(e)}")
                return False, f"Error processing {image_path.name}: {str(e)}"
            
            # Apply post-processing if enabled
            if post_process_mask and not fast_mode:
                try:
                    # Convert to numpy array
                    img_array = np.array(output_image)
                    
                    # Get the alpha channel
                    alpha = img_array[:, :, 3]
                    
                    # Apply stronger edge refinement (increase blur radius)
                    alpha = Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(radius=2.0))
                    alpha = np.array(alpha)
                    
                    # Enhance contrast of the alpha channel
                    alpha = np.clip(alpha * 1.2, 0, 255).astype(np.uint8)
                    
                    # Update the alpha channel
                    img_array[:, :, 3] = alpha
                    
                    # Color decontamination: remove background color spill
                    rgb = img_array[:, :, :3]
                    mask = alpha / 255.0
                    bg_color = np.array([0, 0, 0])  # Assuming black background
                    rgb = (rgb - (1 - mask[..., None]) * bg_color).clip(0, 255).astype(np.uint8)
                    img_array[:, :, :3] = rgb
                    
                    output_image = Image.fromarray(img_array)
                    
                    # Enhance the image
                    enhancer = ImageEnhance.Contrast(output_image)
                    output_image = enhancer.enhance(1.08)  # Slightly stronger contrast
                except Exception as e:
                    logger.warning(f"Post-processing failed for {image_path.name}: {str(e)}")
            
            # Save with optimization
            output_image.save(output_path, 'PNG', optimize=True)
        
        return True, f"Successfully processed {image_path.name}"
        
    except Exception as e:
        return False, f"Error processing {image_path.name}: {str(e)}"

def _needs_alpha_matting(image: Image.Image) -> bool:
    """Determine if an image needs alpha matting based on edge complexity."""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Get the alpha channel
        alpha = img_array[:, :, 3]
        
        # Calculate edge complexity
        edge_complexity = np.std(alpha) / 255.0
        
        # Return True if the image has complex edges
        return edge_complexity > 0.1
    except:
        return False

class BackgroundRemover:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        extensions: List[str] = ['jpg', 'png'],
        prefix: str = 'processed_',
        suffix: str = '',
        num_processes: Optional[int] = None,
        max_size: int = 1500,
        batch_size: int = 10,
        alpha_matting: bool = True,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10,
        post_process_mask: bool = True,
        fast_mode: bool = False
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.extensions = [ext.lower() for ext in extensions]
        self.prefix = prefix
        self.suffix = suffix
        self.num_processes = num_processes or max(1, multiprocessing.cpu_count() - 1)
        self.max_size = max_size
        self.batch_size = batch_size
        self.alpha_matting = alpha_matting
        self.alpha_matting_foreground_threshold = alpha_matting_foreground_threshold
        self.alpha_matting_background_threshold = alpha_matting_background_threshold
        self.alpha_matting_erode_size = alpha_matting_erode_size
        self.post_process_mask = post_process_mask
        self.fast_mode = fast_mode
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_image_files(self) -> List[Path]:
        """Get all image files from the input directory."""
        image_files = []
        for ext in self.extensions:
            image_files.extend(self.input_dir.glob(f"*.{ext}"))
            image_files.extend(self.input_dir.glob(f"*.{ext.upper()}"))
        return sorted(image_files)
    
    def _process_batch(self, batch: List[Path]) -> List[Tuple[bool, str]]:
        """Process a batch of images."""
        return [
            process_single_image(
                image_path,
                self.output_dir,
                self.prefix,
                self.suffix,
                self.max_size,
                self.alpha_matting,
                self.alpha_matting_foreground_threshold,
                self.alpha_matting_background_threshold,
                self.alpha_matting_erode_size,
                self.post_process_mask,
                self.fast_mode
            )
            for image_path in batch
        ]
    
    def process_images(self):
        """Process all images in parallel batches."""
        image_files = self._get_image_files()
        
        if not image_files:
            logger.warning(f"No images found in {self.input_dir} with extensions {self.extensions}")
            return
        
        logger.info(f"Found {len(image_files)} images to process")
        logger.info(f"Using {self.num_processes} processes")
        logger.info(f"Mode: {'Fast' if self.fast_mode else 'Quality'}")
        
        # Split images into batches
        batches = [image_files[i:i + self.batch_size] for i in range(0, len(image_files), self.batch_size)]
        
        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            all_results = []
            for batch_results in tqdm(
                executor.map(self._process_batch, batches),
                total=len(batches),
                desc="Processing batches"
            ):
                all_results.extend(batch_results)
        
        # Log results
        successful = sum(1 for success, _ in all_results if success)
        failed = len(all_results) - successful
        
        logger.info(f"Processing complete: {successful} successful, {failed} failed")
        
        # Log any errors
        for success, message in all_results:
            if not success:
                logger.error(message)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Bulk Background Remover')
    parser.add_argument('--input', required=True, help='Input directory containing images')
    parser.add_argument('--output', required=True, help='Output directory for processed images')
    parser.add_argument('--processes', type=int, help='Number of parallel processes')
    parser.add_argument('--extensions', nargs='+', default=['jpg', 'png'],
                      help='File extensions to process')
    parser.add_argument('--prefix', default='processed_',
                      help='Prefix for output filenames')
    parser.add_argument('--suffix', default='',
                      help='Suffix for output filenames')
    parser.add_argument('--max-size', type=int, default=1500,
                      help='Maximum dimension for resizing (default: 1500)')
    parser.add_argument('--batch-size', type=int, default=10,
                      help='Number of images to process in each batch (default: 10)')
    parser.add_argument('--no-alpha-matting', action='store_false', dest='alpha_matting',
                      help='Disable alpha matting for faster processing')
    parser.add_argument('--foreground-threshold', type=int, default=240,
                      help='Alpha matting foreground threshold (default: 240)')
    parser.add_argument('--background-threshold', type=int, default=10,
                      help='Alpha matting background threshold (default: 10)')
    parser.add_argument('--erode-size', type=int, default=10,
                      help='Alpha matting erode size (default: 10)')
    parser.add_argument('--no-post-process', action='store_false', dest='post_process_mask',
                      help='Disable post-processing mask refinement')
    parser.add_argument('--fast', action='store_true',
                      help='Enable fast mode for quicker processing with slightly reduced quality')
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        remover = BackgroundRemover(
            input_dir=args.input,
            output_dir=args.output,
            extensions=args.extensions,
            prefix=args.prefix,
            suffix=args.suffix,
            num_processes=args.processes,
            max_size=args.max_size,
            batch_size=args.batch_size,
            alpha_matting=args.alpha_matting,
            alpha_matting_foreground_threshold=args.foreground_threshold,
            alpha_matting_background_threshold=args.background_threshold,
            alpha_matting_erode_size=args.erode_size,
            post_process_mask=args.post_process_mask,
            fast_mode=args.fast
        )
        remover.process_images()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 