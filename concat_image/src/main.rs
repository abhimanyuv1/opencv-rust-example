use anyhow::Result;
use opencv::{self as cv, highgui, prelude::*};

fn main() -> Result<()> {
    let img = cv::imgcodecs::imread("lena.jpg", cv::imgcodecs::IMREAD_COLOR).unwrap();
    let mut horiz = Mat::default();
    cv::core::hconcat2(&img, &img, &mut horiz).unwrap();
    highgui::named_window("Horizontal", highgui::WINDOW_GUI_NORMAL).unwrap();
    highgui::imshow("Horizontal", &horiz).unwrap();
    let mut vertical = Mat::default();
    cv::core::vconcat2(&img, &img, &mut vertical).unwrap();
    highgui::named_window("Vertical", highgui::WINDOW_GUI_NORMAL).unwrap();
    highgui::imshow("Vertical", &vertical).unwrap();
    loop {
        let key = highgui::wait_key(1)?;
        if key == 113 {
            // quit with q
            break;
        }
    }

    Ok(())
}
