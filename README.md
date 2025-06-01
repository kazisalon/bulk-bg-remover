# Bulk Background Remover

A Python-based application for batch processing images to remove backgrounds using AI. This tool uses the rembg library (based on U-2-Net) to automatically remove backgrounds from multiple images and save them with transparency.

## Features

- Batch process multiple images from a specified input folder
- Automatic background removal using AI
- Support for JPG and PNG input formats
- Outputs transparent PNG images
- Parallel processing for improved performance
- Error handling and logging
- Customizable file extensions and naming conventions
- Cross-platform support (Windows/Mac/Linux)

## Requirements

- Python 3.8 or higher
- 8GB RAM recommended
- No GPU required, but GPU support available for improved performance

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/bgremover.git
cd bgremover
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python bgremover.py --input input_folder --output output_folder
```

Advanced options:
```bash
python bgremover.py --input input_folder --output output_folder --processes 4 --extensions jpg png
```

### Command Line Arguments

- `--input`: Input folder containing images (required)
- `--output`: Output folder for processed images (required)
- `--processes`: Number of parallel processes (default: number of CPU cores)
- `--extensions`: File extensions to process (default: jpg png)
- `--prefix`: Prefix for output filenames (default: "processed_")
- `--suffix`: Suffix for output filenames (default: "")

## Performance Optimization

- For better performance, consider using a GPU by installing the CUDA version of PyTorch
- Adjust the number of processes based on your CPU cores and available memory
- For large images, consider resizing before processing

## Future Enhancements

- GUI interface using tkinter
- API integration using FastAPI
- Custom AI model support
- Edge refinement options
- Batch size optimization
- Progress tracking and resume capability

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 