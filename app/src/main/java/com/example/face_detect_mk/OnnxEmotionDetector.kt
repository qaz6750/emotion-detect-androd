package com.example.face_detect_mk

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

/**
 * ONNX-based emotion detector for face images
 */
class OnnxEmotionDetector(private val context: Context) {
    
    private companion object {
        const val TAG = "OnnxEmotionDetector"
        const val EMOTION_MODEL_NAME = "fer2013_resnet_emotion.onnx"
    }
    
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var isInitialized = false
    
    /**
     * Initialize the emotion detection model
     */
    fun initialize(): Boolean {
        return try {
            Log.d(TAG, "Initializing emotion detection model...")
            
            // Create ONNX Runtime environment
            ortEnvironment = OrtEnvironment.getEnvironment()
            
            // Load model from assets
            val modelBytes = context.assets.open(EMOTION_MODEL_NAME).use { inputStream ->
                inputStream.readBytes()
            }
            
            Log.d(TAG, "Loaded emotion model: ${modelBytes.size} bytes")
            
            // Create ONNX session
            ortSession = ortEnvironment!!.createSession(modelBytes)
            
            // Log model information
            logModelInfo()
            
            isInitialized = true
            Log.d(TAG, "Emotion detection model initialized successfully")
            
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize emotion detection model", e)
            cleanup()
            false
        }
    }
    
    /**
     * Detect emotion from a face bitmap
     */
    fun detectEmotion(faceBitmap: Bitmap): EmotionData? {
        if (!isInitialized || ortSession == null) {
            Log.e(TAG, "Emotion detector not initialized")
            return null
        }
        
        return try {
            Log.d(TAG, "Detecting emotion from face bitmap: ${faceBitmap.width}x${faceBitmap.height}")
            
            // Preprocess the face image
            val inputBuffer = OnnxEmotionDetectionUtils.preprocessEmotionImage(faceBitmap)
            if (inputBuffer == null) {
                Log.e(TAG, "Failed to preprocess face image for emotion detection")
                return null
            }
            
            // Create ONNX tensor
            val inputShape = longArrayOf(1, 1, 48, 48) // [batch, channels, height, width]
            val inputTensor = OnnxTensor.createTensor(ortEnvironment!!, inputBuffer, inputShape)
            
            Log.d(TAG, "Created emotion input tensor: ${inputShape.contentToString()}")
            
            // Run inference
            val inputs = mapOf("input" to inputTensor)
            val outputs = ortSession!!.run(inputs)
            
            // Extract output
            val outputTensor = outputs[0].value as Array<FloatArray>
            val outputArray = outputTensor[0] // Get first (and only) batch
            
            Log.d(TAG, "Emotion model output shape: [${outputTensor.size}, ${outputArray.size}]")
            Log.d(TAG, "Raw emotion scores: ${outputArray.joinToString { "%.3f".format(it) }}")
            
            // Post-process results
            val emotionResult = OnnxEmotionDetectionUtils.postProcessEmotionOutputs(outputArray)
            
            // Cleanup tensors
            inputTensor.close()
            outputs.close()
            
            emotionResult
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during emotion detection", e)
            null
        }
    }
    
    /**
     * Detect emotion from detected face region in the original image
     */
    fun detectEmotionFromFaceRegion(
        originalBitmap: Bitmap, 
        faceRect: Rect,
        padding: Float = 0.2f
    ): EmotionData? {
        if (!isInitialized) {
            Log.e(TAG, "Emotion detector not initialized")
            return null
        }
        
        return try {
            // Crop face from original image
            val faceBitmap = OnnxEmotionDetectionUtils.cropFaceFromBitmap(
                originalBitmap, faceRect, padding
            )
            
            if (faceBitmap == null) {
                Log.e(TAG, "Failed to crop face from original image")
                return null
            }
            
            Log.d(TAG, "Cropped face region: ${faceBitmap.width}x${faceBitmap.height}")
            
            // Detect emotion from cropped face
            val emotionResult = detectEmotion(faceBitmap)
            
            // Cleanup
            faceBitmap.recycle()
            
            emotionResult
            
        } catch (e: Exception) {
            Log.e(TAG, "Error detecting emotion from face region", e)
            null
        }
    }
    
    /**
     * Detect emotions from multiple face regions
     */
    fun detectEmotionsFromFaceRegions(
        originalBitmap: Bitmap, 
        faceRects: List<Rect>
    ): List<Pair<Rect, EmotionData?>> {
        return faceRects.map { faceRect ->
            val emotion = detectEmotionFromFaceRegion(originalBitmap, faceRect)
            Pair(faceRect, emotion)
        }
    }
    
    /**
     * Check if the detector is initialized
     */
    fun isInitialized(): Boolean {
        return isInitialized
    }
      /**
     * Log model information for debugging
     */
    private fun logModelInfo() {
        try {
            val session = ortSession ?: return
            
            Log.d(TAG, "=== Emotion Model Information ===")
            
            // Input info
            val inputNames = session.inputNames
            Log.d(TAG, "Input names: ${inputNames.joinToString()}")
            
            val inputInfo = session.inputInfo
            for ((inputName, nodeInfo) in inputInfo) {
                Log.d(TAG, "Input '$inputName': $nodeInfo")
            }
            
            // Output info
            val outputNames = session.outputNames
            Log.d(TAG, "Output names: ${outputNames.joinToString()}")
            
            val outputInfo = session.outputInfo
            for ((outputName, nodeInfo) in outputInfo) {
                Log.d(TAG, "Output '$outputName': $nodeInfo")
            }
            
            Log.d(TAG, "==================================")
            
        } catch (e: Exception) {
            Log.w(TAG, "Could not log model info", e)
        }
    }
    
    /**
     * Clean up resources
     */
    fun cleanup() {
        try {
            ortSession?.close()
            // Note: Don't close ortEnvironment as it might be shared
            
            ortSession = null
            isInitialized = false
            
            Log.d(TAG, "Emotion detector resources cleaned up")
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during cleanup", e)
        }
    }
    
    /**
     * Finalize cleanup when object is garbage collected
     */
    protected fun finalize() {
        cleanup()
    }
}
