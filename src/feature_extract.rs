use std::ffi::c_void;
use ndarray::{ArrayBase,
    OwnedRepr,
    prelude::*};
use opencv::{
    boxed_ref::BoxedRef, core::{CV_8U, Mat, Mat_AUTO_STEP, ToInputArray, ToOutputArray}, imgcodecs, prelude::*
};

fn array_to_mat (array: &Array2<u8>) -> Mat {
    let mut standard_layout = array.as_standard_layout();
    let slice = standard_layout.as_slice().unwrap();
    let (height, width) = array.dim();
    
    let mat = unsafe {
        Mat::new_rows_cols_with_data(height as i32, 
            width as i32, 
            slice,
        )
    };
    mat.unwrap().clone_pointee()
}

// pub fn dyalic_gaussian_pyramid(image: &Array2<u8>,
//                                 scale: usize) -> Vec<Array2<u8>> {
//     let mut pyramid_images = Vec::with_capacity(scale);
    
//     let mut clone_image = ndarray_to
    

// }

fn main () {

}

#[cfg(test)]
mod test {
    use  super::*;
    use ndarray::array;

    #[test]
    fn  test_array_dimension() {
        let array= array![
            [10u8, 20u8, 30u8],
            [40u8, 50u8, 60u8],
            [70u8, 80u8, 90u8]
        ];

        let mat = array_to_mat(&array);
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
}