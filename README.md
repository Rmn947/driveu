# DriveU

A Streamlit web application for car rental companies to compare pickup and drop-off vehicle images with automated scratch detection using YOLO and optional AI analysis with Google Gemini.

## Features

- **📤 Image Upload**: Upload car photos with automatic pairing based on naming convention
- **🔄 Auto Pairing**: Automatically pairs pickup and drop-off images by category
- **🔍 YOLO Scratch Detection**: Uses YOLO segmentation model to detect and highlight scratches
- **📊 Side-by-Side Comparison**: View original vs processed images for easy comparison
- **🤖 Gemini AI Analysis**: Optional AI-powered damage assessment comparing pickup and drop-off
- **📱 Responsive Design**: Works on desktop and mobile browsers

## Installation

1. **Clone/Download the project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**: Navigate to `http://localhost:8501`

## Image Naming Convention

Images must follow this naming pattern to be automatically paired:

| Pickup Image | Drop-off Image |
|--------------|----------------|
| `pickup_car_front.jpg` | `drop_car_front.jpg` |
| `pickup_car_rear.jpg` | `drop_car_rear.jpg` |
| `pickup_car_bonnet.jpg` | `drop_car_bonnet.jpg` |
| `pickup_car_boot.jpg` | `drop_car_boot.jpg` |
| `pickup_car_left_front_door.jpg` | `drop_car_left_front_door.jpg` |
| `pickup_car_left_rear_door.jpg` | `drop_car_left_rear_door.jpg` |
| `pickup_car_right_front_door.jpg` | `drop_car_right_front_door.jpg` |
| `pickup_car_right_rear_door.jpg` | `drop_car_right_rear_door.jpg` |
| `pickup_car_tyre_left_front.jpg` | `drop_car_tyre_left_front.jpg` |
| `pickup_car_tyre_left_rear.jpg` | `drop_car_tyre_left_rear.jpg` |
| `pickup_car_tyre_right_front.jpg` | `drop_car_tyre_right_front.jpg` |
| `pickup_car_tyre_right_rear.jpg` | `drop_car_tyre_right_rear.jpg` |
| `pickup_car_front_seats.jpg` | `drop_car_front_seats.jpg` |
| `pickup_car_rear_seats.jpg` | `drop_car_rear_seats.jpg` |
| `pickup_car_odometer.jpg` | `drop_car_odometer.jpg` |
| `pickup_center_console.jpg` | `drop_center_console.jpg` |
| `pickup_glove_box.jpg` | `drop_glove_box.jpg` |

## Usage

### Basic Usage

1. **Upload Images**: Drag and drop or click to upload car images
2. **Automatic Pairing**: Images are automatically paired based on names
3. **View Comparisons**: See side-by-side original images
4. **Process with YOLO**: Click "Process All with YOLO" to detect scratches
5. **Compare Results**: View YOLO-processed images with detected damage highlighted

### Using Gemini AI Analysis

1. **Get API Key**: Obtain a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Enter API Key**: Paste your API key in the sidebar
3. **Analyze**: Click "Analyze with Gemini" on any image pair for AI-powered damage assessment

## Project Structure

```
drive_u/
├── app.py              # Main Streamlit application
├── yolo.py             # Original YOLO script (reference)
├── yolo26-seg.pt       # YOLO segmentation model for scratch detection
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── 0A3671F6/           # Sample images folder
    ├── pickup_car_front.jpg
    ├── drop_car_front.jpg
    └── ...
```

## Configuration

### YOLO Model

The application uses `yolo26-seg.pt` model for scratch detection. Make sure this file is in the project root directory.

### Gemini API

To use Gemini AI analysis:
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Enter the key in the sidebar of the application

## Troubleshooting

### YOLO Model Not Loading
- Ensure `yolo26-seg.pt` is in the same directory as `app.py`
- Install ultralytics: `pip install ultralytics`

### Gemini Analysis Not Working
- Verify your API key is correct
- Install google-generativeai: `pip install google-generativeai`
- Check your API quota limits

### Images Not Pairing
- Verify images follow the naming convention exactly
- Use lowercase for prefixes: `pickup_` and `drop_`
- Ensure no extra characters in filenames

## License

This project is for internal use only.

## Support

For issues or questions, please contact the development team.
