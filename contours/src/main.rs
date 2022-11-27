use anyhow::{Ok, Result};
use opencv::{self as cv, highgui, prelude::*};

fn main() -> Result<()> {
    // Read image
    let image = cv::imgcodecs::imread("iphone.jpg", cv::imgcodecs::IMREAD_COLOR)?;

    // Convert to gray scale
    let mut gray = Mat::default();
    cv::imgproc::cvt_color(&image, &mut gray, cv::imgproc::COLOR_BGR2GRAY, 0)?;

    // 1. Using threshold to find the contours
    // Apply binary thresholding
    let mut threshold_img = Mat::default();
    cv::imgproc::threshold(
        &gray,
        &mut threshold_img,
        150.0,
        255.0,
        cv::imgproc::THRESH_BINARY,
    )?;

    // Find contours
    let mut contours_threshold = cv::types::VectorOfMat::new();
    cv::imgproc::find_contours(
        &threshold_img,
        &mut contours_threshold,
        cv::imgproc::RETR_TREE,
        cv::imgproc::CHAIN_APPROX_SIMPLE,
        cv::core::Point::new(0, 0),
    )?;
    println!(
        "Number of contours found (using threshold method): {}",
        contours_threshold.len()
    );

    // Draw contours
    let hierarchy = Mat::default();
    let mut img_out_threshold = image.clone();
    cv::imgproc::draw_contours(
        &mut img_out_threshold,
        &contours_threshold,
        -1,
        cv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        3,
        cv::imgproc::LINE_AA,
        &hierarchy,
        1,
        cv::core::Point::new(0, 0),
    )?;

    // 2. Using canny edge to find the contour
    let mut edge = Mat::default();
    cv::imgproc::canny(&gray, &mut edge, 30.0, 200.0, 3, true)?;

    // Find contours
    let mut contours_canny = cv::types::VectorOfMat::new();
    cv::imgproc::find_contours(
        &edge,
        &mut contours_canny,
        cv::imgproc::RETR_TREE,
        cv::imgproc::CHAIN_APPROX_SIMPLE,
        cv::core::Point::new(0, 0),
    )?;
    println!(
        "Number of contours found (using canny edge method): {}",
        contours_canny.len()
    );

    // Draw contours
    let hierarchy = Mat::default();
    let mut img_out_canny = image.clone(); // Reset the image copy
    cv::imgproc::draw_contours(
        &mut img_out_canny,
        &contours_canny,
        -1,
        cv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        3,
        cv::imgproc::LINE_AA,
        &hierarchy,
        1,
        cv::core::Point::new(0, 0),
    )?;

    // Combine the image and display it
    // First image is Threshold output and next one is Canny output
    let mut output_img = Mat::default();
    cv::core::hconcat2(&img_out_threshold, &img_out_canny, &mut output_img).unwrap();
    highgui::named_window("Threshold|Canny", highgui::WINDOW_GUI_NORMAL).unwrap();
    highgui::imshow("Threshold|Canny", &output_img).unwrap();

    loop {
        let key = highgui::wait_key(1)?;
        if key == 113 {
            // quit with q
            break;
        }
    }

    Ok(())
}
