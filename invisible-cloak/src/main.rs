use anyhow::{Ok, Result};
use opencv::{self as cv, highgui, prelude::*, videoio};

fn main() -> Result<()> {
    let mut background = Mat::default();
    let mut background_flipped = Mat::default();

    // Create window to display video
    highgui::named_window("windows", highgui::WINDOW_GUI_NORMAL).unwrap();
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY).unwrap();

    // Capture background
    for _ in 0..30 {
        cam.read(&mut background).unwrap();
    }
    cv::core::flip(&background, &mut background_flipped, 1).unwrap();

    loop {
        // Read from the camera
        let mut frame = Mat::default();
        let mut frame_flipped = Mat::default();
        cam.read(&mut frame).unwrap();
        cv::core::flip(&frame, &mut frame_flipped, 1).unwrap();
        // convert to hsv color space
        let mut frame_hsv = Mat::default();
        cv::imgproc::cvt_color(
            &frame_flipped,
            &mut frame_hsv,
            cv::imgproc::COLOR_BGR2HSV,
            0,
        )
        .unwrap();

        // Generate mask to detect red color
        let lowerb = cv::core::Scalar::new(0.0, 120.0, 70.0, 0.0);
        let upperb = cv::core::Scalar::new(10.0, 255.0, 255.0, 0.0);
        let mut mask1 = Mat::default();
        cv::core::in_range(&frame_hsv, &lowerb, &upperb, &mut mask1).unwrap();

        let lowerb = cv::core::Scalar::new(170.0, 120.0, 70.0, 0.0);
        let upperb = cv::core::Scalar::new(180.0, 255.0, 255.0, 0.0);
        let mut mask2 = Mat::default();
        cv::core::in_range(&frame_hsv, &lowerb, &upperb, &mut mask2).unwrap();

        // combine both red color range mask to get final mask
        let mask = mask1 + mask2;

        // Open and dilate the mask image
        let kernel = Mat::ones(3, 3, cv::core::CV_8U);
        let mut mask1 = Mat::default();
        cv::imgproc::morphology_ex(
            &mask.into_result().unwrap(),
            &mut mask1,
            cv::imgproc::MORPH_OPEN,
            &kernel.unwrap(),
            cv::core::Point::new(-1, -1),
            1,
            cv::core::BORDER_CONSTANT,
            cv::imgproc::morphology_default_border_value().unwrap(),
        )
        .unwrap();

        let kernel = Mat::ones(3, 3, cv::core::CV_32F);
        let mut mask2 = Mat::default();
        cv::imgproc::morphology_ex(
            &mask1,
            &mut mask2,
            cv::imgproc::MORPH_DILATE,
            &kernel.unwrap(),
            cv::core::Point::new(-1, -1),
            1,
            cv::core::BORDER_CONSTANT,
            cv::imgproc::morphology_default_border_value().unwrap(),
        )
        .unwrap();

        // Inverted mask to segment out red color
        let mut mask_not = Mat::default();
        cv::core::bitwise_not(&mask2, &mut mask_not, &cv::core::no_array()).unwrap();

        // Segment out red color object from the frame by combining both mask and image
        let mut res1 = Mat::default();
        cv::core::bitwise_and(&frame_flipped, &frame_flipped, &mut res1, &mask_not).unwrap();

        // creating image showing static background frame pixels only for the masked region
        let mut res2 = Mat::default();
        cv::core::bitwise_and(
            &background_flipped,
            &background_flipped,
            &mut res2,
            &mask1,
        )
        .unwrap();

        // Combine both result to get final output
        let mut final_out = Mat::default();
        cv::core::add_weighted(&res1, 1.0, &res2, 1.0, 0.0, &mut final_out, -1).unwrap();

        // Display image
        highgui::imshow("windows", &final_out).unwrap();
        let key = highgui::wait_key(1)?;
        if key == 113 {
            // quit with q
            break;
        }
    }

    Ok(())
}
