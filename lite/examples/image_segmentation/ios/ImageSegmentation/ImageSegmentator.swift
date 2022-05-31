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

class ImageSegmentator {

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var segmenter: ImageSegmenter

  /// Dedicated DispatchQueue for TF Lite operations.
  private let tfLiteQueue: DispatchQueue

  // MARK: - Initialization

  /// Load label list from file.
  private static func loadLabelList() -> [String]? {
    guard
      let labelListPath = Bundle.main.path(
        forResource: Constants.labelsFileName,
        ofType: Constants.labelsFileExtension
      )
    else {
      return nil
    }

    // Parse label list file as JSON.
    do {
      let data = try Data(contentsOf: URL(fileURLWithPath: labelListPath), options: .mappedIfSafe)
      let jsonResult = try JSONSerialization.jsonObject(with: data, options: .mutableLeaves)
      if let labelList = jsonResult as? [String] { return labelList } else { return nil }
    } catch {
      print("Error parsing label list file as JSON.")
      return nil
    }
  }

  /// Create a new Image Segmentator instance.
  static func newInstance(completion: @escaping ((Result<ImageSegmentator>) -> Void)) {
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

      // Construct the path to the label list file.
      guard let labelList = loadLabelList() else {
        print(
          "Failed to load the label list file with name: "
          + "\(Constants.labelsFileName).\(Constants.labelsFileExtension)"
        )
        DispatchQueue.main.async {
          completion(
            .error(
              InitializationError.invalidLabelList(
                "\(Constants.labelsFileName).\(Constants.labelsFileExtension)"
              )))
        }
        return
      }

      // Specify the options for the `ImageSegmenter`.
      let options = ImageSegmenterOptions(modelPath: modelPath)

      do {
        let segmenter = try ImageSegmenter.imageSegmenter(options: options)

        // Create an ImageSegmentator instance and return.
        let segmentator = ImageSegmentator(
          tfLiteQueue: tfLiteQueue,
          segmenter: segmenter,
          labelList: labelList
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
    segmenter: ImageSegmenter,
    labelList: [String]
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
      var startTime: Date = Date()
      var preprocessingTime: TimeInterval = 0
      var inferenceTime: TimeInterval = 0
      var postprocessingTime: TimeInterval = 0
      var visualizationTime: TimeInterval = 0

      do {
        // Preprocessing: Convert the input UIImage to MLImage.
        startTime = Date()
        guard let mlImage = MLImage(image: image) else { return }
        var now = Date()
        preprocessingTime = now.timeIntervalSince(startTime)

        startTime = Date()
        // Segmentation
        segmentationResult = try self.segmenter.segment(gmlImage: mlImage)
        now = Date()
        // Calculate segmentation time.
        inferenceTime = now.timeIntervalSince(startTime)
      } catch let error {
        print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
        DispatchQueue.main.async {
          completion(.error(SegmentationError.internalError(error)))
        }
        return
      }
      startTime = Date()

      /// Postprocessing: Convert `SegmentationResult` to the segmentation mask and color for each pixel.
      guard let parsedOutput = self.parseOutput(segmentationResult: segmentationResult) else { return }

      // Calculate postprocessing time.
      // Note: You may find postprocessing very slow if you run the sample app with Debug build.
      // You will see significant speed up if you rerun using Release build, or change
      // Optimization Level in the project's Build Settings to the same value with Release build.
      var now = Date()
      postprocessingTime = now.timeIntervalSince(startTime)

      startTime = Date()
      // Visualize result into images.
      guard
        let resultImage = ImageSegmentator.imageFromSRGBColorArray(
          pixels: parsedOutput.segmentationImagePixels,
          width: Int(parsedOutput.outputImageSize.width),
          height: Int(parsedOutput.outputImageSize.height)
        ),
        let overlayImage = image.overlayWithImage(image: resultImage, alpha: 0.5)
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
        array: parsedOutput.segmentationMaps,
        resultImage: resultImage,
        overlayImage: overlayImage,
        preprocessingTime: preprocessingTime,
        inferenceTime: inferenceTime,
        postProcessingTime: postprocessingTime,
        visualizationTime: visualizationTime,
        colorLegend: parsedOutput.classColors
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
    let classList = Array(Set(results))
    let classLables = classList.map({ segmentation.coloredLabels[Int($0)].label })
    let classColors: [UIColor] = classList.map({
      let colorLabel = segmentation.coloredLabels[Int($0)]
      return UIColor(red: CGFloat(colorLabel.r)/255.0,
              green: CGFloat(colorLabel.g)/255.0,
              blue: CGFloat(colorLabel.b)/255.0,
              alpha: 0.5)
    })
    // Construct a dictionary of classes found in the image and each class's color used in
    // visualization.
    let classLabelColors = [String: UIColor](uniqueKeysWithValues: zip(classLables, classColors))
    let segmentationImagePixels: [UInt32] = results.map({UInt32(Constants.legendColorList[Int($0)])})
    let twoDimArray = [[UInt8]](repeating: [UInt8](repeating: 0, count: categoryMask.width), count: categoryMask.height)
    var iter = results.makeIterator()
    let newResults: [[UInt8]] = twoDimArray.map { $0.compactMap { _ in iter.next() } }
    return ImageSegmentationParseData(
      segmentationMaps: newResults,
      segmentationImagePixels: segmentationImagePixels,
      classColors: classLabelColors,
      outputImageSize: CGSize(width: categoryMask.width, height: categoryMask.height))
  }

  // MARK: - Utils

  /// Construct an UIImage from a list of sRGB pixels.
  private static func imageFromSRGBColorArray(pixels: [UInt32], width: Int, height: Int) -> UIImage?
  {
    guard width > 0 && height > 0 else { return nil }
    guard pixels.count == width * height else { return nil }

    // Make a mutable copy
    var data = pixels

    // Convert array of pixels to a CGImage instance.
    let cgImage = data.withUnsafeMutableBytes { (ptr) -> CGImage in
      let ctx = CGContext(
        data: ptr.baseAddress,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: MemoryLayout<UInt32>.size * width,
        space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue
        + CGImageAlphaInfo.premultipliedFirst.rawValue
      )!
      return ctx.makeImage()!
    }

    // Convert the CGImage instance to an UIImage instance.
    return UIImage(cgImage: cgImage)
  }
}

// MARK: - Types

/// Representation of the image segmentation result.
struct ImageSegmentationResult {
  /// Segmentation result as an array. Each value represents the most likely class the pixel
  /// belongs to.
  let array: [[UInt8]]

  /// Visualization of the segmentation result.
  let resultImage: UIImage

  /// Overlay the segmentation result on input image.
  let overlayImage: UIImage

  /// Processing time.
  let preprocessingTime: TimeInterval
  let inferenceTime: TimeInterval
  let postProcessingTime: TimeInterval
  let visualizationTime: TimeInterval

  /// Dictionary of classes found in the image, and the color used to represent the class in
  /// segmentation result visualization.
  let colorLegend: [String: UIColor]
}

/// ParseData of the image segmentation result.
struct ImageSegmentationParseData {
  // Masks of secmentation result
  let segmentationMaps: [[UInt8]]
  // Legend Color for each pixel
  let segmentationImagePixels: [UInt32]
  // All class indexs of segmentation result
  let classColors: [String: UIColor]
  // Model output image size
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

  /// List of colors to visualize segmentation result.
  static let legendColorList: [UInt32] = [
    0xFFFF_B300, // Vivid Yellow
    0xFF80_3E75, // Strong Purple
    0xFFFF_6800, // Vivid Orange
    0xFFA6_BDD7, // Very Light Blue
    0xFFC1_0020, // Vivid Red
    0xFFCE_A262, // Grayish Yellow
    0xFF81_7066, // Medium Gray
    0xFF00_7D34, // Vivid Green
    0xFFF6_768E, // Strong Purplish Pink
    0xFF00_538A, // Strong Blue
    0xFFFF_7A5C, // Strong Yellowish Pink
    0xFF53_377A, // Strong Violet
    0xFFFF_8E00, // Vivid Orange Yellow
    0xFFB3_2851, // Strong Purplish Red
    0xFFF4_C800, // Vivid Greenish Yellow
    0xFF7F_180D, // Strong Reddish Brown
    0xFF93_AA00, // Vivid Yellowish Green
    0xFF59_3315, // Deep Yellowish Brown
    0xFFF1_3A13, // Vivid Reddish Orange
    0xFF23_2C16, // Dark Olive Green
    0xFF00_A1C2, // Vivid Blue
  ]
}
