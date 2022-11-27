use anyhow::{Ok, Result};
use opencv::{self as cv, highgui, prelude::*, videoio};

fn main() -> Result<()> {
    highgui::named_window("camera", highgui::WINDOW_GUI_NORMAL).unwrap();
    let mut face_detector = cv::objdetect::CascadeClassifier::new("/opt/homebrew/Cellar/opencv/4.6.0_1/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml").unwrap();
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY).unwrap();

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame).unwrap();

        let mut resize_frame = Mat::default();
        cv::imgproc::resize(
            &frame,
            &mut resize_frame,
            cv::core::Size::new(800, 600),
            0.0,
            0.0,
            cv::imgproc::INTER_LINEAR,
        )
        .unwrap();

        let mut gray = Mat::default();
        cv::imgproc::cvt_color(&resize_frame, &mut gray, cv::imgproc::COLOR_BGR2GRAY, 0).unwrap();

        let mut objects = cv::types::VectorOfRect::new();
        face_detector
            .detect_multi_scale(
                &gray,
                &mut objects,
                1.05,
                5,
                cv::objdetect::CASCADE_SCALE_IMAGE,
                cv::core::Size::new(30, 30),
                cv::core::Size::new(500, 500),
            )
            .unwrap();

        let mut output = resize_frame.clone();
        for rect in objects {
            cv::imgproc::rectangle(
                &mut output,
                rect,
                cv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                cv::imgproc::LINE_4,
                0,
            )
            .unwrap();
        }

        highgui::imshow("camera", &output).unwrap();

        let key = highgui::wait_key(1)?;
        if key == 113 {
            break;
        }
    }
    Ok(())
}
