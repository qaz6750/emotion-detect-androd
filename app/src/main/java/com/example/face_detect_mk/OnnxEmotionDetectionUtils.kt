package com.example.face_detect_mk

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Rect
import android.util.Log
import java.nio.FloatBuffer

/**
 * Emotion detection result data class
 */
data class EmotionData(
    val emotion: String,
    val confidence: Float,
    val allScores: FloatArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as EmotionData

        if (emotion != other.emotion) return false
        if (confidence != other.confidence) return false
        if (!allScores.contentEquals(other.allScores)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = emotion.hashCode()
        result = 31 * result + confidence.hashCode()
        result = 31 * result + allScores.contentHashCode()
        return result
    }
}

/**
 * Utility functions for ONNX emotion detection
 */
object OnnxEmotionDetectionUtils {
    
    private const val TAG = "OnnxEmotionUtils"
    
    // Emotion class labels (based on FER2013 dataset)
    private val EMOTION_LABELS = arrayOf(
        "Angry",     // 0
        "Disgust",   // 1
        "Fear",      // 2
        "Happy",     // 3
        "Sad",       // 4
        "Surprise",  // 5
        "Neutral"    // 6
    )
    
    // Expected input size for emotion model
    const val EMOTION_INPUT_SIZE = 48
    
    /**
     * Crop face region from the original bitmap based on detected face coordinates
     */
    fun cropFaceFromBitmap(
        originalBitmap: Bitmap,
        faceRect: Rect,
        padding: Float = 0.2f
    ): Bitmap? {
        try {
            val width = originalBitmap.width
            val height = originalBitmap.height
            
            // Add padding around the face
            val paddingX = (faceRect.width() * padding).toInt()
            val paddingY = (faceRect.height() * padding).toInt()
            
            // Calculate expanded crop region with bounds checking
            val left = maxOf(0, faceRect.left - paddingX)
            val top = maxOf(0, faceRect.top - paddingY)
            val right = minOf(width, faceRect.right + paddingX)
            val bottom = minOf(height, faceRect.bottom + paddingY)
            
            val cropWidth = right - left
            val cropHeight = bottom - top
            
            if (cropWidth <= 0 || cropHeight <= 0) {
                Log.e(TAG, "Invalid crop dimensions: ${cropWidth}x${cropHeight}")
                return null
            }
            
            Log.d(TAG, "Cropping face region: ($left, $top, $right, $bottom) from ${width}x${height}")
            
            return Bitmap.createBitmap(originalBitmap, left, top, cropWidth, cropHeight)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error cropping face from bitmap", e)
            return null
        }
    }
    
    /**
     * Preprocess face bitmap for emotion recognition
     * Converts to grayscale, resizes to 48x48, and normalizes to [-1, 1]
     */
    fun preprocessEmotionImage(faceBitmap: Bitmap): FloatBuffer? {
        try {
            // Convert to grayscale and resize to 48x48
            val grayscaleBitmap = convertToGrayscale(faceBitmap)
            val resizedBitmap = Bitmap.createScaledBitmap(
                grayscaleBitmap, 
                EMOTION_INPUT_SIZE, 
                EMOTION_INPUT_SIZE, 
                true
            )
            
            Log.d(TAG, "Preprocessed emotion image to ${EMOTION_INPUT_SIZE}x${EMOTION_INPUT_SIZE} grayscale")
            
            // Create float buffer for ONNX input [1, 1, 48, 48]
            val inputSize = 1 * 1 * EMOTION_INPUT_SIZE * EMOTION_INPUT_SIZE
            val buffer = FloatBuffer.allocate(inputSize)
            
            val pixels = IntArray(EMOTION_INPUT_SIZE * EMOTION_INPUT_SIZE)
            resizedBitmap.getPixels(
                pixels, 0, EMOTION_INPUT_SIZE, 
                0, 0, EMOTION_INPUT_SIZE, EMOTION_INPUT_SIZE
            )
            
            // Convert pixels to normalized float values [-1, 1]
            for (pixel in pixels) {
                // Extract grayscale value (R=G=B for grayscale)
                val gray = (pixel shr 16) and 0xFF
                // Normalize from [0, 255] to [-1, 1]
                val normalizedValue = (gray / 255.0f) * 2.0f - 1.0f
                buffer.put(normalizedValue)
            }
            
            buffer.rewind()
            
            // Cleanup
            if (grayscaleBitmap != faceBitmap) {
                grayscaleBitmap.recycle()
            }
            if (resizedBitmap != grayscaleBitmap) {
                resizedBitmap.recycle()
            }
            
            return buffer
            
        } catch (e: Exception) {
            Log.e(TAG, "Error preprocessing emotion image", e)
            return null
        }
    }
    
    /**
     * Convert bitmap to grayscale
     */
    private fun convertToGrayscale(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        
        val grayscaleBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(grayscaleBitmap)
        
        val matrix = Matrix()
        val paint = android.graphics.Paint().apply {
            colorFilter = android.graphics.ColorMatrixColorFilter(
                android.graphics.ColorMatrix().apply {
                    setSaturation(0f) // Convert to grayscale
                }
            )
        }
        
        canvas.drawBitmap(bitmap, matrix, paint)
        
        return grayscaleBitmap
    }
    
    /**
     * Post-process emotion detection outputs
     * Converts model output to emotion classification result
     */
    fun postProcessEmotionOutputs(outputs: FloatArray): EmotionData? {
        try {
            if (outputs.size != EMOTION_LABELS.size) {
                Log.e(TAG, "Invalid output size: ${outputs.size}, expected: ${EMOTION_LABELS.size}")
                return null
            }
            
            // Apply softmax to get probabilities
            val probabilities = applySoftmax(outputs)
            
            // Find the emotion with highest confidence
            var maxIndex = 0
            var maxConfidence = probabilities[0]
            
            for (i in 1 until probabilities.size) {
                if (probabilities[i] > maxConfidence) {
                    maxConfidence = probabilities[i]
                    maxIndex = i
                }
            }
            
            val predictedEmotion = EMOTION_LABELS[maxIndex]
            
            // 详细的情绪概率调试信息
            Log.d(TAG, "=== 情绪检测详细结果 ===")
            for (i in EMOTION_LABELS.indices) {
                val percentage = (probabilities[i] * 100).toInt()
                val marker = if (i == maxIndex) " ← 选中" else ""
                Log.d(TAG, "${EMOTION_LABELS[i]}: ${percentage}%$marker")
            }
            Log.d(TAG, "========================")
            
            Log.d(TAG, "Emotion detection result: $predictedEmotion (${(maxConfidence * 100).toInt()}%)")
            Log.d(TAG, "All emotion scores: ${probabilities.joinToString { "%.2f".format(it) }}")
            
            return EmotionData(
                emotion = predictedEmotion,
                confidence = maxConfidence,
                allScores = probabilities
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Error post-processing emotion outputs", e)
            return null
        }
    }
    
    /**
     * Apply softmax activation to convert logits to probabilities
     */
    private fun applySoftmax(logits: FloatArray): FloatArray {
        // Find max value for numerical stability
        val maxLogit = logits.maxOrNull() ?: 0f
        
        // Compute exp(x - max) for each element
        val expValues = logits.map { kotlin.math.exp((it - maxLogit).toDouble()).toFloat() }
        
        // Compute sum of exp values
        val sumExp = expValues.sum()
        
        // Normalize to get probabilities
        return expValues.map { it / sumExp }.toFloatArray()
    }
    
    /**
     * Get emotion label by index
     */
    fun getEmotionLabel(index: Int): String {
        return if (index in EMOTION_LABELS.indices) {
            EMOTION_LABELS[index]
        } else {
            "Unknown"
        }
    }
    
    /**
     * Get all emotion labels
     */
    fun getAllEmotionLabels(): Array<String> {
        return EMOTION_LABELS.clone()
    }
}
