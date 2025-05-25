package com.example.face_detect_mk

import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import android.view.ViewGroup
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * 实时ONNX人脸检测界面
 */
@androidx.annotation.OptIn(ExperimentalGetImage::class)
@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun RealTimeFaceDetection() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val coroutineScope = rememberCoroutineScope()
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }    // 存储检测到的人脸数据
    var detectedFaces by remember { mutableStateOf<List<FaceData>>(emptyList()) }
    // 人脸检测状态
    var faceCount by remember { mutableStateOf(0) }
    // 帧计数器用于跳帧优化
    var frameCounter by remember { mutableStateOf(0) }
    // 正在处理标志，避免重复处理
    var isProcessing by remember { mutableStateOf(false) }
    // 记录PreviewView的实际显示尺寸
    var previewViewWidth by remember { mutableFloatStateOf(0f) }
    var previewViewHeight by remember { mutableFloatStateOf(0f) }
    // 记录相机图像的原始尺寸
    var cameraImageWidth by remember { mutableFloatStateOf(0f) }
    var cameraImageHeight by remember { mutableFloatStateOf(0f) }
    // 记录相机绑定状态
    var cameraInitialized by remember { mutableStateOf(false) }
    // 错误状态
    var hasError by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf("") }
    // 检测器初始化状态
    var detectorsInitialized by remember { mutableStateOf(false) }

    val cameraPermissionState = rememberPermissionState(
        permission = android.Manifest.permission.CAMERA
    )

    // ONNX Face Detector instance
    val faceDetector = remember { OnnxFaceDetector(context) }

    val previewView = remember {
        PreviewView(context).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    // 初始化ONNX检测器
    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            val faceInitialized = faceDetector.initialize()
            
            withContext(Dispatchers.Main) {
                if (!faceInitialized) {
                    hasError = true
                    errorMessage = "Failed to initialize ONNX face detector"
                } else {
                    detectorsInitialized = true
                    Log.d("RealTimeDetection", "ONNX face detector initialized successfully")
                }
            }
        }
    }

    // 如果权限已授予且检测器已初始化，设置相机
    LaunchedEffect(cameraPermissionState.status.isGranted, detectorsInitialized) {
        if (cameraPermissionState.status.isGranted && detectorsInitialized) {
            try {
                val cameraProvider = cameraProviderFuture.get(2, TimeUnit.SECONDS)

                // 预览用例
                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                // 图像分析用例 - 使用更符合手机屏幕比例的分辨率
                val imageAnalysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(720, 1280))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    // 帧跳跃优化：每三帧处理一次
                    frameCounter++
                    
                    if (frameCounter % 3 == 0 && !isProcessing) {
                        // 每3帧且没有正在处理的任务时才进行检测
                        isProcessing = true
                        analyzeImageWithOnnx(
                            imageProxy = imageProxy,
                            faceDetector = faceDetector,
                            coroutineScope = coroutineScope,
                            onFacesDetected = { faces ->
                                detectedFaces = faces
                                faceCount = faces.size
                                // 记录相机图像的原始尺寸
                                cameraImageWidth = imageProxy.width.toFloat()
                                cameraImageHeight = imageProxy.height.toFloat()
                                isProcessing = false
                            }
                        )
                    } else {
                        // 跳过该帧，使用上一帧的检测结果
                        imageProxy.close()
                    }
                }

                // 解绑之前绑定的用例
                cameraProvider.unbindAll()

                // 将用例绑定到相机
                val camera = cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    imageAnalysis
                )

                cameraInitialized = true
                hasError = false
                errorMessage = ""

            } catch (e: Exception) {
                Log.e("RealTimeFaceDetection", "Camera setup failed: ${e.message}", e)
                hasError = true
                errorMessage = "相机初始化失败: ${e.message}"
            }
        }
    }

    // 清理资源
    DisposableEffect(Unit) {
        onDispose {
            faceDetector.cleanup()
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
        } else if (!detectorsInitialized) {
            Text(
                text = "正在初始化ONNX检测器...",
                color = MaterialTheme.colorScheme.onSurface,
                modifier = Modifier.padding(16.dp)
            )
        } else if (hasError) {
            Text(
                text = errorMessage,
                color = Color.Red,
                modifier = Modifier.padding(16.dp)
            )
        } else {            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(0.7f)
                    .padding(horizontal = 8.dp)
                    .clip(androidx.compose.foundation.shape.RoundedCornerShape(16.dp))
            ) {                // 显示相机预览
                AndroidView(
                    factory = { previewView },
                    modifier = Modifier.fillMaxSize(),
                    update = { view ->
                        // 当PreviewView布局完成后，获取其实际尺寸
                        view.viewTreeObserver.addOnGlobalLayoutListener(object : android.view.ViewTreeObserver.OnGlobalLayoutListener {
                            override fun onGlobalLayout() {
                                if (view.width > 0 && view.height > 0) {
                                    previewViewWidth = view.width.toFloat()
                                    previewViewHeight = view.height.toFloat()
                                    Log.d("RealTimeDetection", "PreviewView actual size: ${previewViewWidth}x${previewViewHeight}")
                                    view.viewTreeObserver.removeOnGlobalLayoutListener(this)
                                }
                            }
                        })
                    }
                )                // 绘制人脸检测框的Canvas覆盖层
                Canvas(modifier = Modifier.fillMaxSize()) {
                    if (previewViewWidth > 0 && previewViewHeight > 0 && 
                        cameraImageWidth > 0 && cameraImageHeight > 0 && 
                        detectedFaces.isNotEmpty()) {
                        
                        Log.d("RealTimeDetection", "Canvas size: ${size.width}x${size.height}")
                        Log.d("RealTimeDetection", "PreviewView size: ${previewViewWidth}x${previewViewHeight}")
                        Log.d("RealTimeDetection", "Camera image size: ${cameraImageWidth}x${cameraImageHeight}")
                        
                        // 检测相机图像是否需要旋转
                        // 如果相机图像是横向(宽>高)但PreviewView是竖向(高>宽)，说明需要旋转90度
                        val needsRotation = (cameraImageWidth > cameraImageHeight) && (previewViewHeight > previewViewWidth)
                        
                        // 根据是否需要旋转来调整图像尺寸用于计算
                        val effectiveImageWidth = if (needsRotation) cameraImageHeight else cameraImageWidth
                        val effectiveImageHeight = if (needsRotation) cameraImageWidth else cameraImageHeight
                        
                        Log.d("RealTimeDetection", "Needs rotation: $needsRotation")
                        Log.d("RealTimeDetection", "Effective image size: ${effectiveImageWidth}x${effectiveImageHeight}")
                        
                        // 计算图像在PreviewView中的实际显示区域
                        val imageAspectRatio = effectiveImageWidth / effectiveImageHeight
                        val previewAspectRatio = previewViewWidth / previewViewHeight
                        
                        val displayedImageWidth: Float
                        val displayedImageHeight: Float
                        val offsetX: Float
                        val offsetY: Float
                        
                        // FILL_CENTER模式下的实际显示计算
                        if (imageAspectRatio > previewAspectRatio) {
                            // 图像更宽，以PreviewView的高度为准，图像的左右会被裁剪
                            displayedImageHeight = previewViewHeight
                            displayedImageWidth = previewViewHeight * imageAspectRatio
                            offsetX = (previewViewWidth - displayedImageWidth) / 2f
                            offsetY = 0f
                        } else {
                            // 图像更高，以PreviewView的宽度为准，图像的上下会被裁剪
                            displayedImageWidth = previewViewWidth
                            displayedImageHeight = previewViewWidth / imageAspectRatio
                            offsetX = 0f
                            offsetY = (previewViewHeight - displayedImageHeight) / 2f
                        }
                        
                        Log.d("RealTimeDetection", "Displayed image size: ${displayedImageWidth}x${displayedImageHeight}")
                        Log.d("RealTimeDetection", "Display offset: ${offsetX}x${offsetY}")
                        
                        // 计算坐标缩放比例
                        val scaleX = displayedImageWidth / effectiveImageWidth
                        val scaleY = displayedImageHeight / effectiveImageHeight
                        
                        Log.d("RealTimeDetection", "Scale factors: scaleX=$scaleX, scaleY=$scaleY")
                        
                        detectedFaces.forEachIndexed { index, face ->
                            Log.d("RealTimeDetection", "Original face $index: left=${face.left}, top=${face.top}, right=${face.right}, bottom=${face.bottom}")
                            
                            val finalLeft: Float
                            val finalTop: Float
                            val finalRight: Float
                            val finalBottom: Float
                            
                            if (needsRotation) {
                                // 需要旋转90度：原来的(x,y)变成(y, imageWidth-x)
                                // 同时考虑前置相机镜像：在旋转后再镜像
                                val rotatedLeft = face.top
                                val rotatedTop = cameraImageWidth - face.right
                                val rotatedRight = face.bottom
                                val rotatedBottom = cameraImageWidth - face.left
                                
                                // 前置相机镜像翻转（在rotated坐标系中）
                                val mirroredLeft = effectiveImageWidth - rotatedRight
                                val mirroredRight = effectiveImageWidth - rotatedLeft
                                
                                finalLeft = mirroredLeft * scaleX + offsetX
                                finalTop = rotatedTop * scaleY + offsetY
                                finalRight = mirroredRight * scaleX + offsetX
                                finalBottom = rotatedBottom * scaleY + offsetY
                                
                                Log.d("RealTimeDetection", "Face $index: rotated=($rotatedLeft, $rotatedTop, $rotatedRight, $rotatedBottom)")
                                Log.d("RealTimeDetection", "Face $index: mirrored=($mirroredLeft, $rotatedTop, $mirroredRight, $rotatedBottom)")
                            } else {
                                // 不需要旋转，只需镜像
                                val mirroredLeft = cameraImageWidth - face.right
                                val mirroredRight = cameraImageWidth - face.left
                                
                                finalLeft = mirroredLeft * scaleX + offsetX
                                finalTop = face.top * scaleY + offsetY
                                finalRight = mirroredRight * scaleX + offsetX
                                finalBottom = face.bottom * scaleY + offsetY
                                
                                Log.d("RealTimeDetection", "Face $index: mirrored=($mirroredLeft, ${face.top}, $mirroredRight, ${face.bottom})")
                            }
                            
                            Log.d("RealTimeDetection", "Face $index: final=($finalLeft, $finalTop, $finalRight, $finalBottom)")
                            
                            // 绘制人脸边框
                            drawRect(
                                color = Color(0xFF00FF00),
                                topLeft = Offset(finalLeft, finalTop),
                                size = androidx.compose.ui.geometry.Size(finalRight - finalLeft, finalBottom - finalTop),
                                style = Stroke(width = 3f)
                            )
                            
                            // 绘制置信度文本
                            val confidenceText = "Face ${index + 1}: %.1f%%".format(face.confidence * 100)
                            drawText(
                                text = confidenceText,
                                x = finalLeft + 5f,
                                y = finalTop - 5f,
                                color = Color(0xFF00FF00)
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = "face counter: $faceCount",
                fontSize = 18.sp,
                color = MaterialTheme.colorScheme.secondary
            )
            
            Text(
                text = "frame counter: $frameCounter",
                fontSize = 14.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
            )
        }
    }
}

// 使用ONNX模型分析图像并检测人脸
@ExperimentalGetImage
private fun analyzeImageWithOnnx(
    imageProxy: ImageProxy,
    faceDetector: OnnxFaceDetector,
    coroutineScope: kotlinx.coroutines.CoroutineScope,
    onFacesDetected: (List<FaceData>) -> Unit
) {
    val mediaImage = imageProxy.image
    if (mediaImage == null) {
        imageProxy.close()
        return
    }

    coroutineScope.launch(Dispatchers.IO) {
        try {
            // 使用工具类将ImageProxy转换为Bitmap
            val bitmap = RealTimeDetectionUtils.imageProxyToBitmap(imageProxy)
            
            if (bitmap == null) {
                Log.e("RealTimeDetection", "Failed to convert ImageProxy to Bitmap")
                withContext(Dispatchers.Main) {
                    onFacesDetected(emptyList())
                }
                return@launch
            }

            Log.d("RealTimeDetection", "Processing bitmap: ${bitmap.width}x${bitmap.height}")

            // 使用ONNX模型检测人脸
            val detectedFaces = faceDetector.detectFaces(bitmap)
            val faceCount = detectedFaces.size

            Log.d("RealTimeDetection", "Detected $faceCount faces")

            // 更新UI
            withContext(Dispatchers.Main) {
                onFacesDetected(detectedFaces)
            }

        } catch (e: Exception) {
            Log.e("RealTimeDetection", "Error processing image: ${e.message}", e)
            withContext(Dispatchers.Main) {
                onFacesDetected(emptyList())
            }
        } finally {
            imageProxy.close()
        }
    }
}

// 四元组数据类
data class Quadruple<A, B, C, D>(val first: A, val second: B, val third: C, val fourth: D)