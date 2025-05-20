package com.example.face_detect_mk

import android.util.Log
import android.util.Size
import android.view.ViewGroup
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.util.concurrent.Executors

/**
 * 实时相机人脸检测界面
 */
@androidx.annotation.OptIn(ExperimentalGetImage::class)
@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun CameraFaceDetection() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    var detectedFaces by remember { mutableStateOf<List<Face>>(emptyList()) }
    
    // 记录分析图像的实际宽高和旋转角度
    var imageWidth by remember { mutableFloatStateOf(640f) }
    var imageHeight by remember { mutableFloatStateOf(480f) }
    var imageRotation by remember { mutableIntStateOf(0) }
    
    val cameraPermissionState = rememberPermissionState(
        permission = android.Manifest.permission.CAMERA
    )
    
    val previewView = remember {
        PreviewView(context).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    
    // 如果权限已授予，设置相机和人脸检测
    LaunchedEffect(cameraPermissionState.status.isGranted) {
        if (cameraPermissionState.status.isGranted) {
            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()
                
                // 设置人脸检测器
                val faceDetectorOptions = FaceDetectorOptions.Builder()
                    .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                    .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
                    .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                    .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                    .setMinFaceSize(0.2f)
                    .enableTracking()
                    .build()
                    
                val faceDetector = FaceDetection.getClient(faceDetectorOptions)
                
                // 图像分析用例
                val imageAnalysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .apply {
                        setAnalyzer(cameraExecutor) { imageProxy ->
                            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                            val image = imageProxy.image
                            
                            if (image != null) {
                                // 记录实际的图像尺寸和旋转角度
                                imageWidth = image.width.toFloat()
                                imageHeight = image.height.toFloat()
                                imageRotation = rotationDegrees
                                
                                val inputImage = InputImage.fromMediaImage(image, rotationDegrees)
                                
                                Log.d("FaceDetection", "Image analysis: width=${image.width}, height=${image.height}, rotation=$rotationDegrees")
                                
                                faceDetector.process(inputImage)
                                    .addOnSuccessListener { faces ->
                                        Log.d("FaceDetection", "Detected ${faces.size} faces, rotation: $rotationDegrees")
                                        faces.forEach { face ->
                                            val bounds = face.boundingBox
                                            Log.d("FaceDetection", "Face bounds: left=${bounds.left}, top=${bounds.top}, " +
                                                    "right=${bounds.right}, bottom=${bounds.bottom}")
                                        }
                                        
                                        // 更新UI显示的人脸列表
                                        detectedFaces = faces
                                    }
                                    .addOnFailureListener { e ->
                                        Log.e("FaceDetection", "Detection failed: ${e.message}")
                                    }
                                    .addOnCompleteListener {
                                        // 完成处理后关闭图像，以便处理下一帧
                                        imageProxy.close()
                                    }
                            } else {
                                imageProxy.close()
                            }
                        }
                    }
                
                // 预览用例
                val preview = Preview.Builder().build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }
                
                try {
                    // 解绑之前绑定的用例
                    cameraProvider.unbindAll()
                    
                    // 将用例绑定到相机
                    val camera = cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_FRONT_CAMERA,  // 使用前置摄像头
                        preview,
                        imageAnalysis
                    )
                } catch (e: Exception) {
                    Log.e("CameraX", "Use case binding failed", e)
                }
                
            }, ContextCompat.getMainExecutor(context))
        }
    }
    
    // 清理资源
    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
        }
    }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        if (!cameraPermissionState.status.isGranted) {
            Button(onClick = { cameraPermissionState.launchPermissionRequest() }) {
                Text("请求相机权限")
            }
        } else {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .border(1.dp, Color.Gray),
                contentAlignment = Alignment.Center
            ) {
                // 相机预览
                AndroidView(
                    factory = { previewView },
                    modifier = Modifier.fillMaxSize()
                )
                
                // 人脸矩形叠加层
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val previewWidth = size.width
                    val previewHeight = size.height
                    
                    detectedFaces.forEach { face ->
                        val rect = face.boundingBox
                        
                        // 计算预览视图与相机分析尺寸的比例
                        val viewAspectRatio = previewWidth / previewHeight
                        val imageAspectRatio = imageWidth / imageHeight
                        
                        // 预览视图的有效尺寸（考虑到可能的纵横比不同）
                        val effectivePreviewWidth: Float
                        val effectivePreviewHeight: Float
                        val xOffset: Float
                        val yOffset: Float
                        
                        if (viewAspectRatio > imageAspectRatio) {
                            // 视图比图像更宽，应该基于高度进行缩放
                            effectivePreviewHeight = previewHeight
                            effectivePreviewWidth = previewHeight * imageAspectRatio
                            xOffset = (previewWidth - effectivePreviewWidth) / 2
                            yOffset = 0f
                        } else {
                            // 视图比图像更窄，应该基于宽度进行缩放
                            effectivePreviewWidth = previewWidth
                            effectivePreviewHeight = previewWidth / imageAspectRatio
                            xOffset = 0f
                            yOffset = (previewHeight - effectivePreviewHeight) / 2
                        }
                        
                        // 计算面部矩形在预览中的位置（考虑到镜像翻转）
                        val widthRatio = effectivePreviewWidth / imageWidth
                        val heightRatio = effectivePreviewHeight / imageHeight
                        
                        // 前置摄像头需要水平翻转
                        val left: Float
                        val right: Float
                        
                        // 处理镜像和旋转问题
                        if (imageRotation == 90 || imageRotation == 270) {
                            // 如果图像旋转了90度或270度，宽高需要互换
                            left = xOffset + translateX(rect.top.toFloat(), effectivePreviewWidth, imageHeight, true)
                            val top = yOffset + translateY(rect.left.toFloat(), effectivePreviewHeight, imageWidth)
                            right = xOffset + translateX(rect.bottom.toFloat(), effectivePreviewWidth, imageHeight, true)
                            val bottom = yOffset + translateY(rect.right.toFloat(), effectivePreviewHeight, imageWidth)
                            
                            drawRect(
                                color = Color.Green,
                                topLeft = Offset(left, top),
                                size = androidx.compose.ui.geometry.Size(right - left, bottom - top),
                                style = Stroke(width = 5f)
                            )
                        } else {
                            // 正常方向或180度旋转
                            left = xOffset + (imageWidth - rect.right) * widthRatio // 镜像翻转X坐标
                            val top = yOffset + rect.top * heightRatio
                            right = xOffset + (imageWidth - rect.left) * widthRatio // 镜像翻转X坐标
                            val bottom = yOffset + rect.bottom * heightRatio
                            
                            drawRect(
                                color = Color.Green,
                                topLeft = Offset(left, top),
                                size = androidx.compose.ui.geometry.Size(right - left, bottom - top),
                                style = Stroke(width = 5f)
                            )
                        }
                    }
                }
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Text(
                text = "检测到 ${detectedFaces.size} 张人脸",
                fontSize = 18.sp,
                color = MaterialTheme.colorScheme.secondary
            )
        }
    }
}
