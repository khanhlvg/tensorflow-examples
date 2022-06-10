// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlowLiteTaskVision
import UIKit

class ImageSegmentationHelper {

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var segmenter: ImageSegmenter

  /// Dedicated DispatchQueue for TF Lite operations.
  private let tfLiteQueue: DispatchQueue

  // MARK: - Initialization

  /// Create a new Image Segmentator instance.
  static func newInstance(completion: @escaping ((Result<ImageSegmentationHelper>) -> Void)) {
    // Create a dispatch queue to ensure all operations on the Intepreter will run serially.
    let tfLiteQueue = DispatchQueue(label: "org.tensorflow.examples.lite.image_segmentation")

    // Run initialization in background thread to avoid UI freeze.
    tfLiteQueue.async {
      // Construct the path to the model file.
      guard
        let modelPath = Bundle.main.path(
          forResource: Constants.modelFileName,
          ofType: Constants.modelFileExtension
        )
      else {
        print(
          "Failed to load the model file with name: "
          + "\(Constants.modelFileName).\(Constants.modelFileExtension)")
        DispatchQueue.main.async {
          completion(
            .error(
              InitializationError.invalidModel(
                "\(Constants.modelFileName).\(Constants.modelFileExtension)"
              )))
        }
        return
      }

      // Specify the options for the `ImageSegmenter`.
      let options = ImageSegmenterOptions(modelPath: modelPath)

      do {
        let segmenter = try ImageSegmenter.segmenter(options: options)

        // Create an ImageSegmentator instance and return.
        let segmentator = ImageSegmentationHelper(
          tfLiteQueue: tfLiteQueue,
          segmenter: segmenter
        )
        DispatchQueue.main.async {
          completion(.success(segmentator))
        }
      } catch let error {
        print("Failed to create the interpreter with error: \(error.localizedDescription)")
        DispatchQueue.main.async {
          completion(.error(InitializationError.internalError(error)))
        }
        return
      }
    }
  }

  /// Initialize Image Segmentator instance.
  fileprivate init(
    tfLiteQueue: DispatchQueue,
    segmenter: ImageSegmenter
  ) {
    // Store TF Lite intepreter
    self.segmenter = segmenter

    // Store the dedicated DispatchQueue for TFLite.
    self.tfLiteQueue = tfLiteQueue
  }

  // MARK: - Image Segmentation

  /// Run segmentation on a given image.
  /// - Parameter image: the target image.
  /// - Parameter completion: the callback to receive segmentation result.
  func runSegmentation(
    _ image: UIImage, completion: @escaping ((Result<ImageSegmentationResult>) -> Void)
  ) {
    tfLiteQueue.async {
      [weak self] in guard let `self` = self else {
        print("The app is in an invalid state.")
        DispatchQueue.main.async {
          completion(.error(SegmentationError.invalidState))
        }
        return
      }
      let segmentationResult: SegmentationResult
      var inferenceTime: TimeInterval = 0
      var postprocessingTime: TimeInterval = 0

      do {
        // Preprocessing: Convert the input UIImage to MLImage.
        let startTime = Date()
        guard let mlImage = MLImage(image: image) else {
          print("The input image is invalid.")
          DispatchQueue.main.async {
            completion(.error(SegmentationError.invalidImage))
          }
          return
        }
        
        // Run segmentation
        segmentationResult = try self.segmenter.segment(mlImage: mlImage)
        
        // Calculate segmentation time.
        inferenceTime = Date().timeIntervalSince(startTime)
      } catch let error {
        print("Failed to invoke TFLite with error: \(error.localizedDescription)")
        DispatchQueue.main.async {
          completion(.error(SegmentationError.internalError(error)))
        }
        return
      }

      /// Postprocessing: Visualize the `SegmentationResult` object.
      let startTime = Date()
      guard let (resultImage, colorLegend) = self.parseOutput(
        segmentationResult: segmentationResult) else {
        print("Failed to parse model output.")
        DispatchQueue.main.async {
          completion(.error(SegmentationError.postProcessingError))
        }
        return
      }
      // Calculate postprocessing time.
      // Note: You may find postprocessing slow when running the sample app with the Debug build.
      // You will see significant speed up if you switch to using Release build, or change
      // Optimization Level in the project's Build Settings to the same value as the Release build.
      postprocessingTime = Date().timeIntervalSince(startTime)

      // Create a representative object that contains the segmentation result.
      let result = ImageSegmentationResult(
        resultImage: resultImage,
        colorLegend: colorLegend,
        inferenceTime: inferenceTime,
        postProcessingTime: postprocessingTime
      )

      // Return the segmentation result.
      DispatchQueue.main.async {
        completion(.success(result))
      }
    }
  }

  // MARK: - Image Segmentation Parse

  /// Run segmentation map and color  for each pixel, if can't get `categoryMask` -> return nil.
  /// - Parameter segmentationResult: The result received from image secmentation process
  private func parseOutput(segmentationResult: SegmentationResult) -> (UIImage, [String: UIColor])? {
    guard let segmentation = segmentationResult.segmentations.first,
          let categoryMask = segmentation.categoryMask else { return nil }
    let mask = categoryMask.mask
    let results = [UInt8](UnsafeMutableBufferPointer(start: mask,
                                                     count: categoryMask.width * categoryMask.height))
    
    // Create a visualization of the segmentation image.
    let alphaChannel: UInt32 = 255
    let classColorsUInt32: [UInt32] = segmentation.coloredLabels.map({
      let colorAsUInt32 = alphaChannel << 24 + // alpha channel
                         UInt32($0.r) << 16 +
                         UInt32($0.g) << 8 +
                         UInt32($0.b)
      return colorAsUInt32
    })
    let segmentationImagePixels: [UInt32] = results.map({ classColorsUInt32[Int($0)] })
    guard let resultImage = UIImage.fromSRGBColorArray(
      pixels: segmentationImagePixels,
      size: CGSize(width: categoryMask.width, height: categoryMask.height)
    ) else { return nil }
    
    // Calculate the list of classes found in the image and its visualization color.
    let classFoundInImageList = IndexSet(Set(results).map({ Int($0) }))
    let filteredColorLabels = classFoundInImageList.map({ segmentation.coloredLabels[$0] })
    let colorLegend = Dictionary<String, UIColor>(uniqueKeysWithValues: filteredColorLabels.map {
      colorLabel in
      let color = UIColor(red: CGFloat(colorLabel.r) / 255.0,
                    green: CGFloat(colorLabel.g) / 255.0,
                    blue: CGFloat(colorLabel.b) / 255.0,
                    alpha: CGFloat(alphaChannel) / 255.0)
      return (colorLabel.label, color)
    })
    
    return (resultImage, colorLegend)
  }
}

// MARK: - Types

/// Representation of the image segmentation result.
struct ImageSegmentationResult {
  /// Visualization of the segmentation result.
  let resultImage: UIImage
  
  /// Dictionary of classes found in the image, and the color used to represent the class in
  /// the segmentation result visualization.
  let colorLegend: [String: UIColor]

  /// Processing time.
  let inferenceTime: TimeInterval
  let postProcessingTime: TimeInterval
}

/// Convenient enum to return result with a callback
enum Result<T> {
  case success(T)
  case error(Error)
}

/// Define errors that could happen in the initialization of this class
enum InitializationError: Error {
  // Invalid TF Lite model
  case invalidModel(String)

  // TF Lite internal Error when initializing
  case internalError(Error)
}

/// Define errors that could happen in when doing image segmentation
enum SegmentationError: Error {
  // The app is in an invalid state
  case invalidState
  
  // Invalid input image
  case invalidImage

  // TF Lite internal Error when running inference
  case internalError(Error)
  
  // Error when processing the TFLite model output
  case postProcessingError

  // Error when visualizing the segmentation result
  case resultVisualizationError
}

// MARK: - Constants
private enum Constants {
  /// The TF Lite segmentation model file
  static let modelFileName = "deeplabv3"
  static let modelFileExtension = "tflite"
}
