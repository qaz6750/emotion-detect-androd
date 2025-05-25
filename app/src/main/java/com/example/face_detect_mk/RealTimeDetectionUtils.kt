package com.example.face_detect_mk

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

/**
 * 实时检测工具类
 * 提供ImageProxy转换、图像绘制等功能
 */
object RealTimeDetectionUtils {
    
    private const val TAG = "RealTimeDetectionUtils"    /**
     * 将ImageProxy转换为Bitmap
     * 使用简化的转换方法
     */
    fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            // 对于JPEG格式，直接解码
            if (imageProxy.format == android.graphics.ImageFormat.JPEG) {
                val buffer = imageProxy.planes[0].buffer
                val bytes = ByteArray(buffer.remaining())
                buffer.get(bytes)
                android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            } else {
                // 对于YUV格式，使用现有的工具方法
                convertYuvToBitmap(imageProxy)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error converting ImageProxy to Bitmap: ${e.message}", e)
            null
        }
    }

    /**
     * YUV转Bitmap的简化方法
     */
    private fun convertYuvToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val image = imageProxy.image ?: return null
            
            // 使用Android的YuvImage进行转换
            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = android.graphics.YuvImage(
                nv21,
                android.graphics.ImageFormat.NV21,
                image.width,
                image.height,
                null
            )

            val out = java.io.ByteArrayOutputStream()
            yuvImage.compressToJpeg(android.graphics.Rect(0, 0, image.width, image.height), 80, out)
            val imageBytes = out.toByteArray()

            android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        } catch (e: Exception) {
            Log.e(TAG, "Error converting YUV to Bitmap: ${e.message}", e)
            null
        }
    }

    /**
     * 在Bitmap上绘制人脸检测框
     * @param originalBitmap 原始位图
     * @param faces 检测到的人脸列表
     * @return 绘制了人脸框的新位图
     */
    fun drawFacesOnBitmap(originalBitmap: Bitmap, faces: List<FaceData>): Bitmap {
        // 创建可变的位图副本
        val mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        // 设置绘制样式
        val paint = Paint().apply {
            color = Color.GREEN
            strokeWidth = 8f
            style = Paint.Style.STROKE
            isAntiAlias = true
        }

        val textPaint = Paint().apply {
            color = Color.GREEN
            textSize = 40f
            isAntiAlias = true
            style = Paint.Style.FILL
        }        // 绘制每个检测到的人脸
        faces.forEachIndexed { index, face ->
            // 绘制人脸边框
            val rect = Rect(face.left, face.top, face.right, face.bottom)
            canvas.drawRect(rect, paint)

            // 绘制置信度文本
            val confidenceText = "Face ${index + 1}: ${(face.confidence * 100).toInt()}%"
            val textX = face.left.toFloat()
            val textY = maxOf(face.top - 10f, textPaint.textSize)
            
            // 绘制文本背景
            val textBounds = Rect()
            textPaint.getTextBounds(confidenceText, 0, confidenceText.length, textBounds)
            val backgroundPaint = Paint().apply {
                color = Color.argb(128, 0, 0, 0) // 半透明黑色背景
                style = Paint.Style.FILL
            }
            canvas.drawRect(
                textX - 5f,
                textY - textBounds.height() - 5f,
                textX + textBounds.width() + 5f,
                textY + 5f,
                backgroundPaint
            )
            
            // 绘制文本
            canvas.drawText(confidenceText, textX, textY, textPaint)

            Log.d(TAG, "Drew face $index: confidence=${(face.confidence * 100).toInt()}%, rect=$rect")
        }

        Log.d(TAG, "Drew ${faces.size} faces on bitmap")
        return mutableBitmap
    }

    /**
     * 镜像翻转位图（用于前置相机）
     */
    fun flipBitmapHorizontally(bitmap: Bitmap): Bitmap {
        val matrix = android.graphics.Matrix().apply {
            preScale(-1.0f, 1.0f)
        }
        return Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, false
        )
    }

    /**
     * 旋转位图
     */
    fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = android.graphics.Matrix().apply {
            postRotate(degrees)
        }
        return Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
        )
    }

    /**
     * 缩放位图到指定尺寸
     */
    fun scaleBitmap(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
    }

    /**
     * 计算两个矩形的重叠面积比（IoU）
     */
    fun calculateIoU(rect1: Rect, rect2: Rect): Float {
        val intersectionLeft = maxOf(rect1.left, rect2.left)
        val intersectionTop = maxOf(rect1.top, rect2.top)
        val intersectionRight = minOf(rect1.right, rect2.right)
        val intersectionBottom = minOf(rect1.bottom, rect2.bottom)

        if (intersectionLeft >= intersectionRight || intersectionTop >= intersectionBottom) {
            return 0f
        }

        val intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop)
        val rect1Area = (rect1.right - rect1.left) * (rect1.bottom - rect1.top)
        val rect2Area = (rect2.right - rect2.left) * (rect2.bottom - rect2.top)
        val unionArea = rect1Area + rect2Area - intersectionArea

        return if (unionArea > 0) intersectionArea.toFloat() / unionArea else 0f
    }

    /**
     * 检查位图是否有效
     */
    fun isValidBitmap(bitmap: Bitmap?): Boolean {
        return bitmap != null && !bitmap.isRecycled && bitmap.width > 0 && bitmap.height > 0
    }

    /**
     * 安全地回收位图
     */
    fun recycleBitmapSafely(bitmap: Bitmap?) {
        try {
            if (bitmap != null && !bitmap.isRecycled) {
                bitmap.recycle()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error recycling bitmap: ${e.message}")
        }
    }
}