package com.example.face_detect_mk

import android.content.Context
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Camera utilities for real-time detection
 */
object CameraUtils {
    
    private const val TAG = "CameraUtils"
    
    /**
     * Setup camera for real-time detection
     */
    fun setupCamera(
        context: Context,
        lifecycleOwner: LifecycleOwner,
        previewView: PreviewView,
        imageAnalyzer: ImageAnalysis.Analyzer,
        cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    ) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                
                // Preview use case
                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }
                
                // Image analysis use case
                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor, imageAnalyzer)
                    }
                
                // Select back camera as default
                val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
                
                try {
                    // Unbind use cases before rebinding
                    cameraProvider.unbindAll()
                    
                    // Bind use cases to camera
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        preview,
                        imageAnalysis
                    )
                    
                    Log.d(TAG, "Camera setup completed successfully")
                    
                } catch (exc: Exception) {
                    Log.e(TAG, "Use case binding failed", exc)
                }
                
            } catch (exc: Exception) {
                Log.e(TAG, "Camera initialization failed", exc)
            }
        }, ContextCompat.getMainExecutor(context))
    }
    
    /**
     * Create image analyzer for real-time detection
     */
    fun createImageAnalyzer(
        onImageAnalyzed: (ImageProxy) -> Unit
    ): ImageAnalysis.Analyzer {
        return ImageAnalysis.Analyzer { imageProxy ->
            try {
                onImageAnalyzed(imageProxy)
            } catch (e: Exception) {
                Log.e(TAG, "Error in image analysis", e)
            } finally {
                imageProxy.close()
            }
        }
    }
    
    /**
     * Check camera permissions
     */
    fun hasCameraPermission(context: Context): Boolean {
        return androidx.core.content.ContextCompat.checkSelfPermission(
            context,
            android.Manifest.permission.CAMERA
        ) == android.content.pm.PackageManager.PERMISSION_GRANTED
    }
    
    /**
     * Get required camera permissions
     */
    fun getCameraPermissions(): Array<String> {
        return arrayOf(android.Manifest.permission.CAMERA)
    }
}
