package com.example.face_detect_mk

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.IOException
import java.nio.FloatBuffer

/**
 * ONNX Face Detector using version-RFB-320 model
 */
class OnnxFaceDetector(private val context: Context) {
    
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var isInitialized = false
    
    companion object {
        private const val TAG = "OnnxFaceDetector"
        // Improve some accuracy
        // private const val MODEL_FILENAME = "version-RFB-320.onnx"
        private const val MODEL_FILENAME = "version-RFB-320.onnx"
        private const val INPUT_NAME = "input"
        private const val OUTPUT_SCORES_NAME = "scores"
        private const val OUTPUT_BOXES_NAME = "boxes"
    }
    
    /**
     * Initialize the ONNX Runtime session
     */
    fun initialize(): Boolean {
        return try {
            Log.d(TAG, "Initializing ONNX Face Detector...")
            
            // Create ONNX Runtime environment
            ortEnvironment = OrtEnvironment.getEnvironment()
            
            // Load model from assets
            val modelBytes = loadModelFromAssets()
            if (modelBytes == null) {
                Log.e(TAG, "Failed to load model from assets")
                return false
            }
            
            // Create session
            ortSession = ortEnvironment?.createSession(modelBytes)
            
            if (ortSession != null) {
                isInitialized = true
                Log.d(TAG, "ONNX Face Detector initialized successfully")
                
                // Log model info
                logModelInfo()
                return true
            } else {
                Log.e(TAG, "Failed to create ONNX session")
                return false
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing ONNX Face Detector: ${e.message}", e)
            cleanup()
            false
        }
    }
    
    /**
     * Detect faces in the given bitmap
     */
    fun detectFaces(bitmap: Bitmap): List<FaceData> {
        if (!isInitialized || ortSession == null || ortEnvironment == null) {
            Log.e(TAG, "ONNX Face Detector not initialized")
            return emptyList()
        }
        
        return try {
            Log.d(TAG, "Starting face detection on bitmap ${bitmap.width}x${bitmap.height}")            // Preprocess image
            val inputData = OnnxFaceDetectionUtils.preprocessImage(bitmap)
            
            // Create input tensor using FloatBuffer
            val inputShape = longArrayOf(1, 3, 240, 320)
            val inputBuffer = FloatBuffer.wrap(inputData)
            val inputTensor = OnnxTensor.createTensor(ortEnvironment!!, inputBuffer, inputShape)
            
            // Run inference
            val inputs = mapOf(INPUT_NAME to inputTensor)
            val outputs = ortSession!!.run(inputs)
            
            // Extract outputs
            val scoresOutput = outputs.get(OUTPUT_SCORES_NAME)?.get()?.value as Array<Array<FloatArray>>?
            val boxesOutput = outputs.get(OUTPUT_BOXES_NAME)?.get()?.value as Array<Array<FloatArray>>?
            
            if (scoresOutput == null || boxesOutput == null) {
                Log.e(TAG, "Failed to get model outputs")
                return emptyList()
            }
              // Convert outputs to flat arrays
            val scores = scoresOutput[0].flatMap { it.toList() }.toFloatArray()
            val boxes = boxesOutput[0].flatMap { it.toList() }.toFloatArray()
            
            Log.d(TAG, "Model outputs - Scores: ${scores.size}, Boxes: ${boxes.size}")
            
            // Debug: print first few detections
            for (i in 0 until minOf(5, scores.size/2)) {
                val conf = scores[i * 2 + 1]
                if (conf > 0.1f) { // Low threshold for debugging
                    Log.d(TAG, "Detection $i: confidence=$conf, box=[${boxes[i*4]}, ${boxes[i*4+1]}, ${boxes[i*4+2]}, ${boxes[i*4+3]}]")
                }
            }
            
            // Post-process results
            val detections = OnnxFaceDetectionUtils.postProcessOutputs(
                scores, boxes, bitmap.width, bitmap.height
            )
            
            // Cleanup tensors
            inputTensor.close()
            outputs.forEach { it.value?.close() }
            
            Log.d(TAG, "Face detection completed, found ${detections.size} faces")
            detections
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during face detection: ${e.message}", e)
            emptyList()
        }
    }
    
    /**
     * Load ONNX model from assets or models folder
     */
    private fun loadModelFromAssets(): ByteArray? {
        return try {
            // First try to load from models folder
            val modelsDir = context.filesDir.parentFile?.resolve("models")
            val modelFile = modelsDir?.resolve(MODEL_FILENAME)
            
            if (modelFile?.exists() == true) {
                Log.d(TAG, "Loading model from models folder: ${modelFile.absolutePath}")
                return modelFile.readBytes()
            }
            
            // Fallback to assets folder
            Log.d(TAG, "Loading model from assets folder")
            context.assets.open(MODEL_FILENAME).use { inputStream ->
                inputStream.readBytes()
            }
        } catch (e: IOException) {
            Log.e(TAG, "Error loading model file: ${e.message}", e)
            null
        }
    }
    
    /**
     * Log model information for debugging
     */
    private fun logModelInfo() {
        try {
            ortSession?.let { session ->
                Log.d(TAG, "Model input names: ${session.inputNames}")
                Log.d(TAG, "Model output names: ${session.outputNames}")
                
                session.inputInfo.forEach { (name, info) ->
                    Log.d(TAG, "Input '$name': ${info.info}")
                }
                
                session.outputInfo.forEach { (name, info) ->
                    Log.d(TAG, "Output '$name': ${info.info}")
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not log model info: ${e.message}")
        }
    }
    
    /**
     * Clean up resources
     */
    fun cleanup() {
        try {
            ortSession?.close()
            ortEnvironment?.close()
            ortSession = null
            ortEnvironment = null
            isInitialized = false
            Log.d(TAG, "ONNX Face Detector cleanup completed")
        } catch (e: Exception) {
            Log.e(TAG, "Error during cleanup: ${e.message}", e)
        }
    }
    
    /**
     * Check if detector is initialized
     */
    fun isInitialized(): Boolean = isInitialized
}