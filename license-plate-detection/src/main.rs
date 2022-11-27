use anyhow::Result;
use opencv::{self as cv, highgui, prelude::*};

fn display_img(img: Mat, name: &str) {
    highgui::named_window(name, highgui::WINDOW_AUTOSIZE).unwrap();
    highgui::imshow(name, &img).unwrap();
    //highgui::wait_keyq(0).unwrap();
}

fn main() -> Result<()> {
    let img = cv::imgcodecs::imread("licenseplate.jpg", cv::imgcodecs::IMREAD_COLOR).unwrap();
    let mut resize_img = Mat::default();
    cv::imgproc::resize(
        &img,
        &mut resize_img,
        cv::core::Size::new(300, 300),
        0.0,
        0.0,
        cv::imgproc::INTER_NEAREST,
    )
    .unwrap();

    display_img(resize_img.clone(), "resized");

    let mut gray = Mat::default();
    cv::imgproc::cvt_color(&resize_img, &mut gray, cv::imgproc::COLOR_BGR2GRAY, 0).unwrap();

    let mut blur = Mat::default();
    cv::imgproc::gaussian_blur(
        &gray,
        &mut blur,
        cv::core::Size::new(3, 3),
        0.0,
        0.0,
        cv::core::BORDER_DEFAULT,
    )
    .unwrap();

    let mut edges = Mat::default();
    cv::imgproc::canny(&blur, &mut edges, 10.0, 200.0, 3, false).unwrap();

    display_img(edges.clone(), "canny");

    let mut contours = cv::types::VectorOfMat::new();
    cv::imgproc::find_contours(
        &edges,
        &mut contours,
        cv::imgproc::RETR_LIST,
        cv::imgproc::CHAIN_APPROX_SIMPLE,
        cv::core::Point::new(0, 0),
    )
    .unwrap();

    let mut cnts = contours.to_vec();
    cnts.sort_by(|a, b| {
        let i = f64::abs(cv::imgproc::contour_area(a, false).unwrap());
        let j = f64::abs(cv::imgproc::contour_area(b, false).unwrap());
        return j.partial_cmp(&i).unwrap();
    });

    let mut screen_cnts = Mat::default();
    for c in &cnts {
        let perimeter = cv::imgproc::arc_length(&c, true).unwrap();
        let mut approx = Mat::default();
        cv::imgproc::approx_poly_dp(&c, &mut approx, 0.018 * perimeter, true).unwrap();
        if approx.rows() == 4 {
            screen_cnts = approx.clone();
            break;
        }
    }

    let mut img1 = resize_img.clone();
    cv::imgproc::draw_contours(
        &mut img1,
        &screen_cnts,
        -1,
        cv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        3,
        0,
        &cv::core::no_array(),
        1,
        cv::core::Point::new(0, 0),
    )
    .unwrap();

    display_img(img1.clone(), "rectange");

    let bound_rect = cv::imgproc::bounding_rect(&screen_cnts).unwrap();
    let mut img1 = resize_img.clone();
    cv::imgproc::rectangle(
        &mut img1,
        bound_rect,
        cv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        cv::imgproc::LINE_8,
        0,
    )
    .unwrap();

    display_img(img1.clone(), "license_plate");

    // crop image only license plate
    let cropped_img = cv::core::Mat::roi(&resize_img, bound_rect).unwrap();

    display_img(cropped_img.clone(), "cropped");

    // Save to file and detect text from file
    //cv::imgcodecs::imwrite("cropped.png", &cropped_img, &cv::core::Vector::default()).unwrap();
    //let result = tesseract::ocr("cropped.png", "eng").unwrap();
    //println!("Result of Image : {}", result);

    // From directly data
    // ?? How to pass Null to the below API ??
    let mut ocr = <dyn cv::text::OCRTesseract>::create(
        "/opt/homebrew/Cellar/tesseract/5.2.0/share/tessdata/",
        "eng",
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
        cv::text::OEM_CUBE_ONLY,
        cv::text::PSM_AUTO,
    )
    .unwrap();

    let result = cv::text::OCRTesseract::run(&mut ocr, &cropped_img, 50, 0).unwrap();
    println!("Result of OCR: {}", result);

    highgui::wait_key(0).unwrap();

    Ok(())
}
