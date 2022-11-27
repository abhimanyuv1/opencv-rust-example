use anyhow::{Ok, Result};
use opencv::{self as cv, highgui, prelude::*, videoio};

fn main() -> Result<()> {
    let image = cv::imgcodecs::imread("qrcode.png", cv::imgcodecs::IMREAD_COLOR).unwrap();
    let mut grey = Mat::default();
    cv::imgproc::cvt_color(&image, &mut grey, cv::imgproc::COLOR_BGR2GRAY, 0).unwrap();

    // Using opencv QR detector
    let mut qr_detector = cv::objdetect::QRCodeDetector::default().unwrap();
    let mut rect = Mat::default();
    let mut rectified_image = Mat::default();
    let result = qr_detector
        .detect_and_decode(&grey, &mut rect, &mut rectified_image)
        .unwrap();
    if result.len() > 0 {
        println!("Decoded data: {}", String::from_utf8(result).unwrap());
    }

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

        let mut rect = Mat::default();
        let mut rectified_image = Mat::default();
        let result = qr_detector
            .detect_and_decode(&resize_frame, &mut rect, &mut rectified_image)
            .unwrap();
        if result.len() > 0 {
            println!("Decoded data: {}", String::from_utf8(result).unwrap());
        }

        highgui::imshow("camera", &frame).unwrap();

        let key = highgui::wait_key(1)?;
        if key == 113 {
            break;
        }
    }

    Ok(())
}
