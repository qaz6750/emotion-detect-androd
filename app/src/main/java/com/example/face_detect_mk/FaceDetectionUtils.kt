package com.example.face_detect_mk

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import android.widget.Toast
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions

/**
 * 从Uri加载位图
 */
fun getBitmapFromUri(context: Context, uri: Uri): Bitmap? {
    return try {
        val inputStream = context.contentResolver.openInputStream(uri)
        BitmapFactory.decodeStream(inputStream)
    } catch (e: Exception) {
        Log.e("FaceDetection", "Error loading bitmap: ${e.message}")
        null
    }
}

/**
 * 在图片中检测人脸
 */
fun detectFaces(context: Context, bitmap: Bitmap, onComplete: (List<Face>) -> Unit) {
    // 高精度人脸检测配置
    val options = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
        .setMinFaceSize(0.15f)
        .enableTracking()
        .build()

    val faceDetector = FaceDetection.getClient(options)
    val image = InputImage.fromBitmap(bitmap, 0)
    
    faceDetector.process(image)
        .addOnSuccessListener { faces ->
            onComplete(faces)
            if (faces.isEmpty()) {
                Toast.makeText(context, "没有检测到人脸", Toast.LENGTH_SHORT).show()
            }
        }
        .addOnFailureListener { e ->
            Log.e("FaceDetection", "Face detection failed: ${e.message}")
            Toast.makeText(context, "人脸检测失败: ${e.message}", Toast.LENGTH_SHORT).show()
            onComplete(emptyList())
        }
}

/**
 * X坐标转换函数，处理镜像问题
 */
fun translateX(
    x: Float,
    previewWidth: Float,
    imageWidth: Float,
    isMirrored: Boolean
): Float {
    if (isMirrored) {
        return previewWidth - (x / imageWidth * previewWidth)
    }
    return x / imageWidth * previewWidth
}

/**
 * Y坐标转换函数
 */
fun translateY(
    y: Float,
    previewHeight: Float,
    imageHeight: Float
): Float {
    return y / imageHeight * previewHeight
}
