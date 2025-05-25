package com.example.face_detect_mk

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

/**
 * 后置摄像头实时检测工具类 - 简化版本
 */
object RealTimeBackDetectionUtils {

    /**
     * 将ImageProxy转换为Bitmap
     */
    fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val image = imageProxy.image ?: return null
            
            when (imageProxy.format) {
                ImageFormat.YUV_420_888 -> {
                    // YUV格式转换为Bitmap
                    val yBuffer = image.planes[0].buffer
                    val uBuffer = image.planes[1].buffer  
                    val vBuffer = image.planes[2].buffer

                    val ySize = yBuffer.remaining()
                    val uSize = uBuffer.remaining()
                    val vSize = vBuffer.remaining()

                    val nv21 = ByteArray(ySize + uSize + vSize)

                    // U和V是交错的，我们需要先拷贝Y
                    yBuffer.get(nv21, 0, ySize)
                    
                    // 然后拷贝V和U（交错）
                    val uvPixelStride = image.planes[1].pixelStride
                    if (uvPixelStride == 1) {
                        uBuffer.get(nv21, ySize, uSize)
                        vBuffer.get(nv21, ySize + uSize, vSize)
                    } else {
                        // 处理UV交错的情况
                        var pos = ySize
                        for (i in 0 until uSize step uvPixelStride) {
                            nv21[pos++] = vBuffer.get(i)
                            nv21[pos++] = uBuffer.get(i)
                        }
                    }

                    val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
                    val out = ByteArrayOutputStream()
                    yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
                    val imageBytes = out.toByteArray()
                    BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                }
                
                ImageFormat.JPEG -> {
                    // JPEG格式直接解码
                    val buffer = image.planes[0].buffer
                    val bytes = ByteArray(buffer.remaining())
                    buffer.get(bytes)
                    BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                }
                
                else -> {
                    Log.w("RealTimeBackDetectionUtils", "不支持的图像格式: ${imageProxy.format}")
                    null
                }
            }
        } catch (e: Exception) {
            Log.e("RealTimeBackDetectionUtils", "图像转换失败: ${e.message}", e)
            null
        }
    }    /**
     * 转换人脸坐标从相机图像到Canvas绘制坐标（后置摄像头修复版本）
     */
    fun transformFaceCoordinates(
        faces: List<FaceData>,
        cameraImageWidth: Float,
        cameraImageHeight: Float,
        previewViewWidth: Float,
        previewViewHeight: Float,
        canvasWidth: Float,
        canvasHeight: Float
    ): List<FaceData> {
        if (faces.isEmpty() || cameraImageWidth <= 0 || cameraImageHeight <= 0 ||
            previewViewWidth <= 0 || previewViewHeight <= 0 ||
            canvasWidth <= 0 || canvasHeight <= 0) {
            return emptyList()
        }

        Log.d("RealTimeBackDetectionUtils", "坐标转换参数:")
        Log.d("RealTimeBackDetectionUtils", "  相机图像: ${cameraImageWidth}x${cameraImageHeight}")
        Log.d("RealTimeBackDetectionUtils", "  PreviewView: ${previewViewWidth}x${previewViewHeight}")
        Log.d("RealTimeBackDetectionUtils", "  Canvas: ${canvasWidth}x${canvasHeight}")

        return faces.mapIndexed { index, face ->
            Log.d("RealTimeBackDetectionUtils", "原始人脸 $index: (${face.left}, ${face.top}, ${face.right}, ${face.bottom})")
              // 步骤1：处理后置摄像头的坐标系问题
            // 
            // 原理解释：
            // 1. 后置摄像头的物理传感器安装角度与前置摄像头不同
            // 2. 当用户竖屏持握设备时，后置摄像头传感器相对于屏幕坐标系旋转了90度
            // 3. 这导致传感器的X轴对应屏幕的Y轴，传感器的Y轴对应屏幕的-X轴
            // 
            // 坐标变换：
            // 传感器坐标 (sensor_x, sensor_y) -> 屏幕坐标 (screen_x, screen_y)
            // screen_x = sensor_height - sensor_y  (270度旋转的X分量)
            // screen_y = sensor_x                  (270度旋转的Y分量)
            //
            // 人脸框变换：
            // left = height - bottom    (左边界 = 图像高度 - 原底边界)
            // top = left               (上边界 = 原左边界)  
            // right = height - top     (右边界 = 图像高度 - 原上边界)
            // bottom = right           (下边界 = 原右边界)
            
            val needsRotation = kotlin.math.abs(cameraImageWidth - cameraImageHeight) < 100f
            
            val adjustedFace = if (needsRotation) {
                // 应用270度旋转变换：(x,y) -> (height-y, x)
                // 这个变换确保用户的左右移动对应框的左右移动，上下移动对应框的上下移动
                FaceData(
                    left = (cameraImageHeight - face.bottom).toInt(),    // 新左边界
                    top = face.left,                                     // 新上边界
                    right = (cameraImageHeight - face.top).toInt(),      // 新右边界  
                    bottom = face.right,                                 // 新下边界
                    confidence = face.confidence
                )
            } else {
                face
            }
            
            // 使用调整后的相机图像尺寸
            val adjustedCameraWidth = if (needsRotation) cameraImageHeight else cameraImageWidth
            val adjustedCameraHeight = if (needsRotation) cameraImageWidth else cameraImageHeight
            
            Log.d("RealTimeBackDetectionUtils", "旋转后人脸 $index: (${adjustedFace.left}, ${adjustedFace.top}, ${adjustedFace.right}, ${adjustedFace.bottom})")
            Log.d("RealTimeBackDetectionUtils", "调整后相机尺寸: ${adjustedCameraWidth}x${adjustedCameraHeight}")
            
            // 步骤2：计算PreviewView中图像的实际显示区域
            val imageAspectRatio = adjustedCameraWidth / adjustedCameraHeight
            val previewAspectRatio = previewViewWidth / previewViewHeight
            
            val actualDisplayWidth: Float
            val actualDisplayHeight: Float
            val offsetX: Float
            val offsetY: Float
            
            if (imageAspectRatio > previewAspectRatio) {
                // 图像更宽，受PreviewView宽度限制
                actualDisplayWidth = previewViewWidth
                actualDisplayHeight = previewViewWidth / imageAspectRatio
                offsetX = 0f
                offsetY = (previewViewHeight - actualDisplayHeight) / 2f
            } else {
                // 图像更高，受PreviewView高度限制
                actualDisplayWidth = previewViewHeight * imageAspectRatio
                actualDisplayHeight = previewViewHeight
                offsetX = (previewViewWidth - actualDisplayWidth) / 2f
                offsetY = 0f
            }
            
            Log.d("RealTimeBackDetectionUtils", "  实际显示区域: ${actualDisplayWidth}x${actualDisplayHeight}, 偏移: (${offsetX}, ${offsetY})")
            
            // 步骤3：将人脸坐标从调整后的相机图像坐标转换到PreviewView坐标
            val scaleX = actualDisplayWidth / adjustedCameraWidth
            val scaleY = actualDisplayHeight / adjustedCameraHeight
            
            val previewLeft = adjustedFace.left * scaleX + offsetX
            val previewTop = adjustedFace.top * scaleY + offsetY
            val previewRight = adjustedFace.right * scaleX + offsetX
            val previewBottom = adjustedFace.bottom * scaleY + offsetY
            
            Log.d("RealTimeBackDetectionUtils", "  PreviewView坐标: (${previewLeft}, ${previewTop}, ${previewRight}, ${previewBottom})")
            
            // 步骤4：从PreviewView坐标转换到Canvas坐标
            val canvasScaleX = canvasWidth / previewViewWidth
            val canvasScaleY = canvasHeight / previewViewHeight
            
            val canvasLeft = previewLeft * canvasScaleX
            val canvasTop = previewTop * canvasScaleY
            val canvasRight = previewRight * canvasScaleX
            val canvasBottom = previewBottom * canvasScaleY
            
            Log.d("RealTimeBackDetectionUtils", "  最终Canvas坐标: (${canvasLeft}, ${canvasTop}, ${canvasRight}, ${canvasBottom})")
            
            // 返回转换后的人脸数据
            FaceData(
                left = canvasLeft.toInt(),
                top = canvasTop.toInt(),
                right = canvasRight.toInt(),
                bottom = canvasBottom.toInt(),
                confidence = face.confidence
            )
        }
    }
}
