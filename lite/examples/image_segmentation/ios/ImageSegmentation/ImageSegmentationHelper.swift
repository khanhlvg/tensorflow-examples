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
      let segmentationResult: SegmentationResult
      var startTime = Date()
      var now = Date()
      var inferenceTime: TimeInterval = 0
      var postprocessingTime: TimeInterval = 0
      var visualizationTime: TimeInterval = 0

      do {
        // Preprocessing: Convert the input UIImage to MLImage.
        startTime = Date()
        guard let mlImage = MLImage(image: image) else { return }
        
        // Segmentation
        segmentationResult = try self.segmenter.segment(mlImage: mlImage)
        
        // Calculate segmentation time.
        now = Date()
        inferenceTime = now.timeIntervalSince(startTime)
      } catch let error {
        print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
        DispatchQueue.main.async {
          completion(.error(SegmentationError.internalError(error)))
        }
        return
      }

      /// Postprocessing: Convert `SegmentationResult` to the segmentation mask and color for each pixel.
      startTime = Date()
      guard let parsedOutput = self.parseOutput(segmentationResult: segmentationResult) else {
        print("Failed to parse model output.")
        DispatchQueue.main.async {
          completion(.error(SegmentationError.postProcessingError))
        }
        return
      }

      // Calculate postprocessing time.
      // Note: You may find postprocessing very slow if you run the sample app with Debug build.
      // You will see significant speed up if you rerun using Release build, or change
      // Optimization Level in the project's Build Settings to the same value with Release build.
      postprocessingTime = now.timeIntervalSince(startTime)

      // Visualize result into images.
      startTime = Date()
      guard
        let overlayImage = image.overlayWithImage(image: parsedOutput.resultImage, alpha: 0.5)
      else {
        print("Failed to visualize segmentation result.")
        DispatchQueue.main.async {
          completion(.error(SegmentationError.resultVisualizationError))
        }
        return
      }

      // Calculate visualization time.
      now = Date()
      visualizationTime = now.timeIntervalSince(startTime)

      // Create a representative object that contains the segmentation result.
      let result = ImageSegmentationResult(
        resultImage: parsedOutput.resultImage,
        overlayImage: overlayImage,
        inferenceTime: inferenceTime,
        postProcessingTime: postprocessingTime,
        visualizationTime: visualizationTime,
        colorLegend: parsedOutput.colorLegend
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
  private func parseOutput(segmentationResult: SegmentationResult) -> ImageSegmentationParseData? {
    guard let segmentation = segmentationResult.segmentations.first,
          let categoryMask = segmentation.categoryMask else { return nil }
    let mask = categoryMask.mask
    let results = [UInt8](UnsafeMutableBufferPointer(start: mask, count: categoryMask.width * categoryMask.height))
    
    // Create a visualization of the segmentation image.
    let alphaChannel: UInt32 = 255
    let classColorsUInt32: [UInt32] = segmentation.coloredLabels.map({
      let colorAsUInt = alphaChannel << 24 + // alpha channel
                         UInt32($0.r) << 16 +
                         UInt32($0.g) << 8 +
                         UInt32($0.b)
      return colorAsUInt
    })
    let segmentationImagePixels: [UInt32] = results.map({ classColorsUInt32[Int($0)] })
    guard let resultImage = UIImage.fromSRGBColorArray(
      pixels: segmentationImagePixels,
      size: CGSize(width: categoryMask.width, height: categoryMask.height)
    ) else { return nil }
    
    // Calculate the list of classes found in the image and its visualization color.
    let classList = IndexSet(Set(results).map({ Int($0) }))
    let filteredColorLabels = classList.map({ segmentation.coloredLabels[$0] })
    let colorLegend = Dictionary<String, UIColor>(uniqueKeysWithValues: filteredColorLabels.map {
      colorLabel in
      let color = UIColor(red: CGFloat(colorLabel.r) / 255.0,
                    green: CGFloat(colorLabel.g) / 255.0,
                    blue: CGFloat(colorLabel.b) / 255.0,
                    alpha: CGFloat(alphaChannel) / 255.0)
      return (colorLabel.label, color)
    })
    
    return ImageSegmentationParseData(
      resultImage: resultImage,
      colorLegend: colorLegend,
      outputImageSize: CGSize(width: categoryMask.width, height: categoryMask.height))
  }
}

// MARK: - Types

/// Representation of the image segmentation result.
struct ImageSegmentationResult {
  /// Visualization of the segmentation result.
  let resultImage: UIImage

  /// Overlay the segmentation result on input image.
  let overlayImage: UIImage

  /// Processing time.
  let inferenceTime: TimeInterval
  let postProcessingTime: TimeInterval
  let visualizationTime: TimeInterval

  /// Dictionary of classes found in the image, and the color used to represent the class in
  /// segmentation result visualization.
  let colorLegend: [String: UIColor]
}

/// ParseData of the image segmentation result.
struct ImageSegmentationParseData {
  /// Legend color for each pixel
  let resultImage: UIImage
  
  /// Dictionary of classes found in the image, and the color used to represent the class in
  /// segmentation result visualization.
  let colorLegend: [String: UIColor]
  
  /// Model output image size
  let outputImageSize: CGSize
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

  // Invalid label list
  case invalidLabelList(String)

  // TF Lite Internal Error when initializing
  case internalError(Error)
}

/// Define errors that could happen in when doing image segmentation
enum SegmentationError: Error {
  // Invalid input image
  case invalidImage

  // TF Lite Internal Error when initializing
  case internalError(Error)
  
  // Error when processing the TFLite model output
  case postProcessingError

  // Invalid input image
  case resultVisualizationError
}

// MARK: - Constants
private enum Constants {
  /// Label list that the segmentation model detects.
  static let labelsFileName = "deeplabv3_labels"
  static let labelsFileExtension = "json"

  /// The TF Lite segmentation model file
  static let modelFileName = "deeplabv3"
  static let modelFileExtension = "tflite"
}
