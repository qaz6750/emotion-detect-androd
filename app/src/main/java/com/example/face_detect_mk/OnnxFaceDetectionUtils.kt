package com.example.face_detect_mk

import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.net.Uri
import android.util.Log
import java.io.InputStream
import kotlin.math.max
import kotlin.math.min

/**
 * Face detection result data class
 */
data class FaceData(
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int,
    val confidence: Float
) {
    fun toRect(): Rect {
        return Rect(left, top, right, bottom)
    }
}

/**
 * Utility functions for ONNX face detection
 */
object OnnxFaceDetectionUtils {
    
    private const val TAG = "OnnxFaceDetectionUtils"
    private const val TARGET_WIDTH = 320
    private const val TARGET_HEIGHT = 240
    
    /**
     * Convert URI to Bitmap
     */
    fun getBitmapFromUri(context: Context, uri: Uri): Bitmap? {
        return try {
            val inputStream: InputStream? = context.contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()
            bitmap
        } catch (e: Exception) {
            Log.e(TAG, "Error converting URI to Bitmap: ${e.message}")
            null
        }
    }
    
    /**
     * Preprocess bitmap for ONNX model input
     * Resize to 320x240 and normalize to [-1, 1] range
     */
    fun preprocessImage(bitmap: Bitmap): FloatArray {
        // Resize bitmap to target size
        val resizedBitmap = resizeBitmap(bitmap, TARGET_WIDTH, TARGET_HEIGHT)
        
        // Convert to RGB and normalize
        val pixels = IntArray(TARGET_WIDTH * TARGET_HEIGHT)
        resizedBitmap.getPixels(pixels, 0, TARGET_WIDTH, 0, 0, TARGET_WIDTH, TARGET_HEIGHT)
        
        // Convert to NCHW format: [1, 3, 240, 320] and normalize to [-1, 1]
        val inputData = FloatArray(1 * 3 * TARGET_HEIGHT * TARGET_WIDTH)
        
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            
            // Normalize: (pixel - 127) / 128 to get [-1, 1] range
            val rNorm = (r - 127f) / 128f
            val gNorm = (g - 127f) / 128f
            val bNorm = (b - 127f) / 128f
            
            // NCHW format: channels first
            val y = i / TARGET_WIDTH
            val x = i % TARGET_WIDTH
            
            inputData[y * TARGET_WIDTH + x] = rNorm  // R channel
            inputData[TARGET_HEIGHT * TARGET_WIDTH + y * TARGET_WIDTH + x] = gNorm  // G channel
            inputData[2 * TARGET_HEIGHT * TARGET_WIDTH + y * TARGET_WIDTH + x] = bNorm  // B channel
        }
        
        return inputData
    }
    
    /**
     * Resize bitmap to specified dimensions
     */
    private fun resizeBitmap(bitmap: Bitmap, width: Int, height: Int): Bitmap {
        val matrix = Matrix()
        val scaleX = width.toFloat() / bitmap.width
        val scaleY = height.toFloat() / bitmap.height
        matrix.setScale(scaleX, scaleY)
        
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, false)
    }
    
    /**
     * Post-process ONNX model outputs to get face detections
     * @param scores: [1, 4420, 2] confidence scores
     * @param boxes: [1, 4420, 4] bounding boxes in center format
     * @param originalWidth: original image width for coordinate scaling
     * @param originalHeight: original image height for coordinate scaling
     * @param confidenceThreshold: minimum confidence threshold (default 0.7)
     * @return List of face detections
     */
    fun postProcessOutputs(
        scores: FloatArray,
        boxes: FloatArray,
        originalWidth: Int,
        originalHeight: Int,
        confidenceThreshold: Float = 0.7f
    ): List<FaceData> {
        val detections = mutableListOf<FaceData>()
        val numDetections = 4420
          // Extract valid detections above threshold
        for (i in 0 until numDetections) {
            val confidence = scores[i * 2 + 1]  // Use positive class score
            
            if (confidence > confidenceThreshold) {
                // Get box coordinates (normalized corner format: x1, y1, x2, y2)
                val x1Norm = boxes[i * 4]
                val y1Norm = boxes[i * 4 + 1]
                val x2Norm = boxes[i * 4 + 2]
                val y2Norm = boxes[i * 4 + 3]
                
                // Convert normalized coordinates to pixel coordinates
                val left = (x1Norm * originalWidth).toInt()
                val top = (y1Norm * originalHeight).toInt()
                val right = (x2Norm * originalWidth).toInt()
                val bottom = (y2Norm * originalHeight).toInt()
                
                // Clamp coordinates to image bounds
                val clampedLeft = max(0, left)
                val clampedTop = max(0, top)
                val clampedRight = min(originalWidth, right)
                val clampedBottom = min(originalHeight, bottom)
                
                // Add debug logging
                Log.d(TAG, "Detection $i: confidence=$confidence, " +
                      "normalized=($x1Norm, $y1Norm, $x2Norm, $y2Norm), " +
                      "pixel=($clampedLeft, $clampedTop, $clampedRight, $clampedBottom)")
                
                detections.add(FaceData(clampedLeft, clampedTop, clampedRight, clampedBottom, confidence))
            }
        }
        
        Log.d(TAG, "Found ${detections.size} faces before NMS")
        
        // Apply Non-Maximum Suppression
        return applyNMS(detections, 0.4f)
    }
    
    /**
     * Apply Non-Maximum Suppression to remove overlapping detections
     */
    private fun applyNMS(detections: List<FaceData>, iouThreshold: Float): List<FaceData> {
        if (detections.isEmpty()) return emptyList()
        
        // Sort by confidence (descending)
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val kept = mutableListOf<FaceData>()
        val suppressed = BooleanArray(sortedDetections.size) { false }
        
        for (i in sortedDetections.indices) {
            if (suppressed[i]) continue
            
            kept.add(sortedDetections[i])
            
            // Suppress overlapping detections
            for (j in i + 1 until sortedDetections.size) {
                if (suppressed[j]) continue
                
                val iou = calculateIoU(sortedDetections[i], sortedDetections[j])
                if (iou > iouThreshold) {
                    suppressed[j] = true
                }
            }
        }
        
        Log.d(TAG, "Kept ${kept.size} faces after NMS")
        return kept
    }
    
    /**
     * Calculate Intersection over Union (IoU) between two face detections
     */
    private fun calculateIoU(face1: FaceData, face2: FaceData): Float {
        val intersectLeft = max(face1.left, face2.left)
        val intersectTop = max(face1.top, face2.top)
        val intersectRight = min(face1.right, face2.right)
        val intersectBottom = min(face1.bottom, face2.bottom)
        
        if (intersectLeft >= intersectRight || intersectTop >= intersectBottom) {
            return 0f
        }
        
        val intersectArea = (intersectRight - intersectLeft) * (intersectBottom - intersectTop)
        val area1 = (face1.right - face1.left) * (face1.bottom - face1.top)
        val area2 = (face2.right - face2.left) * (face2.bottom - face2.top)
        val unionArea = area1 + area2 - intersectArea
        
        return if (unionArea > 0) intersectArea.toFloat() / unionArea else 0f
    }
}