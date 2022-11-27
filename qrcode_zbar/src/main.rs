use anyhow::{Ok, Result};
use cv::imgproc::LINE_4;
use image::{DynamicImage, GenericImageView, GrayImage};
use opencv::{self as cv, highgui, prelude::*, videoio};
use rayon::prelude::*;
use zbar_rust::ZBarImageScanner;

fn main() -> Result<()> {
    let mut scanner = ZBarImageScanner::new();
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY).unwrap();

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame).unwrap();
        let mut gray = Mat::default();
        cv::imgproc::cvt_color(&frame, &mut gray, cv::imgproc::COLOR_BGR2GRAY, 0).unwrap();

        let mut rgbim = GrayImage::new(
            gray.cols().try_into().unwrap(),
            gray.rows().try_into().unwrap(),
        );
        let data = gray.data_bytes()?;
        (&mut *rgbim)
            .par_chunks_mut(1)
            .zip(data.par_chunks(1))
            .for_each(|(rgbim_pix, mat_pix)| {
                let b = mat_pix[0];
                rgbim_pix[0] = b;
            });
        let dynimage = DynamicImage::ImageLuma8(rgbim);

        let (width, height) = dynimage.dimensions();
        let luma_img = dynimage.as_luma8().unwrap();

        let results = scanner.scan_y800(luma_img.to_vec(), width, height).unwrap();
        let mut loc_x: Vec<i32> = vec![];
        let mut loc_y: Vec<i32> = vec![];
        let mut loc_size: usize = 0;

        for result in results {
            println!("{}", String::from_utf8(result.data).unwrap());
            loc_size = result.loc_size;
            loc_x = result.loc_x;
            loc_y = result.loc_y;
        }

        for i in 0..loc_size {
            cv::imgproc::line(
                &mut frame,
                cv::core::Point::new(loc_x[i], loc_y[i]),
                cv::core::Point::new(loc_x[(i + 1) % loc_size], loc_y[(i + 1) % loc_size]),
                cv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                3,
                LINE_4,
                0,
            )
            .unwrap();
        }

        highgui::imshow("camera", &frame).unwrap();

        let key = highgui::wait_key(1)?;
        if key == 113 {
            break;
        }
    }
    Ok(())
}
