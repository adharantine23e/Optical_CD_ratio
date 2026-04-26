use std::ffi::c_void;
use std::collections::HashMap;
use std::sync::atomic::{Ordering};
use ndarray::{prelude::*};
use opencv::{
    imgproc,    
    core::{ CV_8UC3, CV_8UC1, Mat, Mat_AUTO_STEP, split, Size, Vector, AlgorithmHint, bitwise_and}, 
    prelude::*
};
use fast_slic_rust::arrays::LABImage;
use fast_slic_rust::common::Config;
use fast_slic_rust::slic::{iterate, Clusters};

enum ArrayInput<'a> {
    TwoDim(&'a Array2<u8>),
    ThreeDim(&'a Array3<u8>)
}

enum ImageFormat {
    BGR,
    HSV
}

struct ImageChannels {
    image_format: ImageFormat,
    first_channel: Mat,
    second_channel: Mat,
    third_channel: Mat
}

impl ImageChannels {
    fn split_channel(image_mat: &Mat, image_format: ImageFormat) -> opencv::Result<Self> {
        let mut channels = Vector::<Mat>::new();
        split(image_mat, &mut channels);

        Ok(
            ImageChannels { 
                 image_format: image_format,
                 first_channel: channels.get(0).unwrap(),
                 second_channel: channels.get(1).unwrap(),
                 third_channel: channels.get(2).unwrap()
                }
        )
    }
}

#[allow(dead_code)]
fn apply_histogram_equalization(image_mat: &Mat) -> opencv::Result<Mat> {
    let mut clahe = imgproc::create_clahe(3.0, 
                                                    Size::new(8, 8)).unwrap();
    let mut dst = Mat::default();
    clahe.apply(image_mat, &mut dst)?;
    Ok(dst)
}

#[allow(dead_code)]
fn calculate_histogram(image_mat: &Mat, mask: &Mat) -> opencv::Result<Mat> {
    let mut hist = Mat::default();
    let channels: Vector<i32> = Vector::from(vec![0]);
    let hist_size: Vector<i32> = Vector::from(vec![256]);
    let ranges: Vector<f32> = Vector::from(vec![0.0, 256.0]);

    imgproc::calc_hist(image_mat, 
                         &channels, 
                         mask, 
                         &mut hist, 
                         &hist_size, 
                         &ranges,
                        false);
    Ok(hist)
}

#[allow(dead_code)]
fn segment_hist_calculation(slic_segments: Mat, image_chosen: Mat) -> Result<Vec<Vec<f32>>, opencv::Error> {
    if slic_segments.rows() != image_chosen.rows() || slic_segments.cols() != image_chosen.cols() {
        return Err(opencv::Error::new(opencv::core::StsBadArg, 
                                        "Mismatch shape between slic_segments and image_chosen"));
    } else {
        let rows = slic_segments.rows();
        let cols = slic_segments.cols();
        let mut histogram_value: Vec<Vec<f32>> = Vec::new();
        let mut slic_unique_segments = HashMap::new();
        for row in 0..rows {
            for col in 0..cols {
                let segment_id = *slic_segments.at_2d::<u8>(row, col).unwrap();
                *slic_unique_segments.entry(segment_id).or_insert(0) += 1;
            }
        }

        for (_, (seg_val, _))in slic_unique_segments.iter().enumerate() {
            let mut zero_mat = Mat::zeros(rows, cols, CV_8UC1).unwrap().to_mat().unwrap();
            for row in 0..rows {
                for col in 0..cols {
                    let segment_id = *slic_segments.at_2d::<u8>(row, col).unwrap();
                    if segment_id == *seg_val {
                        *zero_mat.at_2d_mut::<u8>(row, col).unwrap() = 255;
                        
                        // Get the segment area based on the mask
                        let mut masked_image = Mat::default();
                        bitwise_and(&image_chosen,
                                     &image_chosen, 
                                     &mut masked_image,
                                    &zero_mat).unwrap();
                        let equalized_masked_image = apply_histogram_equalization(&masked_image).unwrap();
                        let hist_segment = calculate_histogram(&equalized_masked_image, 
                                                                    &Mat::default()).unwrap()
                                                                                        .reshape(0, 1)
                                                                                        .unwrap()
                                                                                        .clone_pointee();                                            
                        
                        histogram_value.push(hist_segment.data_typed::<f32>().unwrap().to_vec());
                    }
                }
            }

        }

        Ok(histogram_value)
    }
}

#[allow(dead_code)]
fn array_to_mat<'a> (array: ArrayInput) -> opencv::Result<Mat> {
    match array {
        ArrayInput::TwoDim(arr) => {
            let mut standard_layout = arr.as_standard_layout();
            let slice = standard_layout.as_slice().unwrap();
            let (height, width) = arr.dim();
            
            let mat = unsafe {
                Mat::new_rows_cols_with_data(
                    height as i32, 
                    width as i32, 
                    slice,
                )?
            }.clone_pointee();
            
            Ok(mat)
        }
        ArrayInput::ThreeDim(arr) => {
            let (height, width, channels) = arr.dim();
            if channels != 3 {
                return Err(opencv::Error::new(
                    opencv::core::StsBadArg,
                    "Array set to ThreeDim but channel is not 3",
                ));

            }
            let slice  = arr.as_slice().ok_or_else(|| opencv::Error::new(opencv::core::StsBadArg,
                                                            "Image is not contigous"))?;
            let mat= unsafe {
                Mat::new_rows_cols_with_data_unsafe(
                    height as i32, 
                    width as i32, 
                    CV_8UC3, 
                    slice.as_ptr() as *mut c_void, 
                    Mat_AUTO_STEP)?
                    .into()
            };
            Ok(mat)
        }
    }
}


pub fn feature_extraction (image_mat: Mat, n_segments: i32, compactness: i32) {
    let height = image_mat.rows();
    let width = image_mat.cols();
    // Convert BGR to HSV
    let mut hsv_format = Mat::default();
    imgproc::cvt_color(&image_mat,
        &mut hsv_format, 
        imgproc::COLOR_BGR2HSV, 
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT);

    // Split channels 
    let bgr = ImageChannels::split_channel(&image_mat, ImageFormat::BGR).unwrap();
    let hsv = ImageChannels::split_channel(&hsv_format, ImageFormat::HSV).unwrap();
    
    // Convert image_mat from Mat to &[u8] and LABImage for fast SLIC
    let mut rgb_image_mat = Mat::default();
    imgproc::cvt_color(&image_mat, 
                        &mut rgb_image_mat, 
                        imgproc::COLOR_BGR2RGB, 
                        0, 
                        AlgorithmHint::ALGO_HINT_DEFAULT);
    let data_byte: &[u8] = rgb_image_mat.data_bytes().unwrap();
    let lab_image = LABImage::from_srgb(data_byte, 
                                                 width as usize, 
                                                 height as usize);
    // Set up config value
    let mut config = Config::default();
    config.num_of_clusters = n_segments as u16;
    config.compactness = compactness as f32;

    let mut cluster = Clusters::initialize_clusters(&lab_image, &config);
    iterate(&lab_image, &config, &mut cluster);
    
    // Get the label
    let slic_label: Vec<u8> = cluster.assignments
                        .data
                        .iter()
                        .map(|x| x.load(Ordering::Relaxed))
                        .map(|x| x as u8)
                        .collect();

    let slic_label_mat = unsafe {
                Mat::new_rows_cols_with_data(
                    height as i32, 
                    width as i32, 
                    slic_label.as_slice(),
        ).unwrap()
    }.clone_pointee();



}   

fn main () {

}

#[cfg(test)]
mod test {
    use  super::*;
    use ndarray::array;
    use opencv::core::Vec3b;

    #[test]
    fn  test_array_2_d_dimension() {
        let array= array![
            [10u8, 20u8, 30u8],
            [40u8, 50u8, 60u8],
            [70u8, 80u8, 90u8]
        ];

        let mat = array_to_mat(ArrayInput::TwoDim(&array)).unwrap();
        assert_eq!(mat.rows(), 3);
        assert_eq!(mat.cols(), 3);

        // Verify pixel values
        assert_eq!(*mat.at_2d::<u8>(0, 0).unwrap(), 10u8);
        assert_eq!(*mat.at_2d::<u8>(0, 1).unwrap(), 20u8);
        assert_eq!(*mat.at_2d::<u8>(0, 2).unwrap(), 30u8);
        assert_eq!(*mat.at_2d::<u8>(1, 0).unwrap(), 40u8);
        assert_eq!(*mat.at_2d::<u8>(1, 1).unwrap(), 50u8);
        assert_eq!(*mat.at_2d::<u8>(1, 2).unwrap(), 60u8);
    }

    #[test]
    fn test_array_3_d_dimension () {
        let array = array![
            [[10u8, 20u8, 30u8], [1u8, 2u8, 3u8]],
            [[40u8, 50u8, 60u8], [4u8, 5u8, 6u8]]
        ];
        let mat = array_to_mat(ArrayInput::ThreeDim(&array)).unwrap();
        let pixel_00 = mat.at_2d::<Vec3b>(0, 0).unwrap();
        let pixel_01= mat.at_2d::<Vec3b>(0, 1).unwrap();
        let pixel_10 = mat.at_2d::<Vec3b>(1, 0).unwrap();
        assert_eq!(mat.rows(), 2);
        assert_eq!(mat.cols(), 2);
        assert_eq!(mat.channels(), 3);

        assert_eq!(pixel_00[0], 10u8);
        assert_eq!(pixel_00[1], 20u8);
        assert_eq!(pixel_00[2], 30u8);

        assert_eq!(pixel_01[0], 1u8);
        assert_eq!(pixel_01[1], 2u8);
        assert_eq!(pixel_01[2], 3u8);

        assert_eq!(pixel_10[0], 40u8);
        assert_eq!(pixel_10[1], 50u8);
        assert_eq!(pixel_10[2], 60u8);



    }
}