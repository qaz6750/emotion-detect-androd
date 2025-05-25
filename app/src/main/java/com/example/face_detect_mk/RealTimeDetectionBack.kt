package com.example.face_detect_mk

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
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
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
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

/**
 * 后置摄像头实时人脸检测界面 - 简化版本，仅检测人脸位置
 */
@androidx.annotation.OptIn(ExperimentalGetImage::class)
@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun RealTimeBackFaceDetection(
    onBack: () -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val coroutineScope = rememberCoroutineScope()
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    // 状态变量
    var detectedFaces by remember { mutableStateOf<List<FaceData>>(emptyList()) }
    var faceCount by remember { mutableStateOf(0) }
    var previewViewWidth by remember { mutableFloatStateOf(0f) }
    var previewViewHeight by remember { mutableFloatStateOf(0f) }
    var cameraImageWidth by remember { mutableFloatStateOf(0f) }
    var cameraImageHeight by remember { mutableFloatStateOf(0f) }
    var hasError by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf("") }
    var detectorsInitialized by remember { mutableStateOf(false) }

    val cameraPermissionState = rememberPermissionState(
        permission = android.Manifest.permission.CAMERA
    )

    // 人脸检测器
    val faceDetector = remember { OnnxFaceDetector(context) }
    val previewView = remember {
        PreviewView(context).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FIT_CENTER  // 改为FIT_CENTER以保持宽高比
        }
    }

    // 初始化人脸检测器
    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            val initialized = faceDetector.initialize()
            
            withContext(Dispatchers.Main) {
                if (!initialized) {
                    hasError = true
                    errorMessage = "人脸检测器初始化失败"
                } else {
                    detectorsInitialized = true
                    Log.d("RealTimeBackDetection", "人脸检测器初始化成功")
                }
            }
        }
    }

    // 设置相机
    LaunchedEffect(cameraPermissionState.status.isGranted, detectorsInitialized) {
        if (cameraPermissionState.status.isGranted && detectorsInitialized) {
            try {
                val cameraProvider = ProcessCameraProvider.getInstance(context).get()

                // 预览用例
                val preview = Preview.Builder().build()
                preview.setSurfaceProvider(previewView.surfaceProvider)

                // 图像分析用例
                val imageAnalysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(1280, 720))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

                imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    analyzeImage(
                        imageProxy = imageProxy,
                        faceDetector = faceDetector,
                        coroutineScope = coroutineScope
                    ) { faces ->
                        detectedFaces = faces
                        faceCount = faces.size
                        cameraImageWidth = imageProxy.width.toFloat()
                        cameraImageHeight = imageProxy.height.toFloat()
                    }
                }

                // 绑定到后置摄像头
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalysis
                )

                hasError = false
                errorMessage = ""

            } catch (e: Exception) {
                Log.e("RealTimeBackDetection", "相机设置失败: ${e.message}", e)
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

    // UI界面
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {
        when {
            !cameraPermissionState.status.isGranted -> {
                // 权限请求界面
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    FloatingActionButton(
                        onClick = { cameraPermissionState.launchPermissionRequest() }
                    ) {
                        Text("请求相机权限")
                    }
                }
            }
            
            !detectorsInitialized -> {
                // 初始化中界面
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "正在初始化检测器...",
                        color = Color.White,
                        style = MaterialTheme.typography.headlineSmall
                    )
                }
            }
            
            hasError -> {
                // 错误界面
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = errorMessage,
                        color = Color.Red,
                        style = MaterialTheme.typography.headlineSmall
                    )
                }
            }
            
            else -> {
                // 相机预览界面
                AndroidView(
                    factory = { previewView },
                    modifier = Modifier.fillMaxSize(),
                    update = { view ->
                        view.viewTreeObserver.addOnGlobalLayoutListener(object : android.view.ViewTreeObserver.OnGlobalLayoutListener {
                            override fun onGlobalLayout() {
                                if (view.width > 0 && view.height > 0) {
                                    previewViewWidth = view.width.toFloat()
                                    previewViewHeight = view.height.toFloat()
                                    view.viewTreeObserver.removeOnGlobalLayoutListener(this)
                                }
                            }
                        })
                    }
                )

                // 绘制人脸框
                Canvas(modifier = Modifier.fillMaxSize()) {
                    if (previewViewWidth > 0 && previewViewHeight > 0 && 
                        cameraImageWidth > 0 && cameraImageHeight > 0 && 
                        detectedFaces.isNotEmpty()) {
                        
                        val transformedFaces = RealTimeBackDetectionUtils.transformFaceCoordinates(
                            faces = detectedFaces,
                            cameraImageWidth = cameraImageWidth,
                            cameraImageHeight = cameraImageHeight,
                            previewViewWidth = previewViewWidth,
                            previewViewHeight = previewViewHeight,
                            canvasWidth = size.width,
                            canvasHeight = size.height
                        )
                          transformedFaces.forEachIndexed { index, face ->
                            // 绘制人脸边框
                            drawRect(
                                color = Color.Green,
                                topLeft = Offset(face.left.toFloat(), face.top.toFloat()),
                                size = androidx.compose.ui.geometry.Size(
                                    (face.right - face.left).toFloat(),
                                    (face.bottom - face.top).toFloat()
                                ),
                                style = Stroke(width = 4f)
                            )

                            // 绘制人脸标签
                            val originalFace = detectedFaces[index]
                            val confidenceText = "Face ${index + 1}: %.1f%%".format(originalFace.confidence * 100)
                              drawContext.canvas.nativeCanvas.drawText(
                                confidenceText,
                                face.left.toFloat() + 10f,
                                face.top.toFloat() - 10f,
                                android.graphics.Paint().apply {
                                    color = android.graphics.Color.GREEN
                                    textSize = 40f
                                    isAntiAlias = true
                                }
                            )
                        }
                    }
                }
            }
        }

        // 关闭按钮
        FloatingActionButton(
            onClick = onBack,
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(16.dp)
                .size(56.dp),
            shape = CircleShape,
            containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.8f)
        ) {
            Icon(
                imageVector = Icons.Default.Close,
                contentDescription = "关闭",
                tint = MaterialTheme.colorScheme.onSurface
            )
        }

        // 检测状态信息
        if (faceCount > 0) {
            Text(
                text = "检测到 $faceCount 张人脸",
                color = Color.White,
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier
                    .align(Alignment.BottomStart)
                    .padding(16.dp)
                    .background(
                        Color.Black.copy(alpha = 0.6f),
                        androidx.compose.foundation.shape.RoundedCornerShape(8.dp)
                    )
                    .padding(horizontal = 12.dp, vertical = 8.dp)
            )
        }
    }
}

// 简化的图像分析函数
@ExperimentalGetImage
private fun analyzeImage(
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
            // 转换为Bitmap
            val bitmap = RealTimeBackDetectionUtils.imageProxyToBitmap(imageProxy)
            
            if (bitmap == null) {
                Log.e("RealTimeBackDetection", "Failed to convert ImageProxy to Bitmap")
                withContext(Dispatchers.Main) {
                    onFacesDetected(emptyList())
                }
                return@launch
            }

            // 检测人脸
            val detectedFaces = faceDetector.detectFaces(bitmap)
            Log.d("RealTimeBackDetection", "检测到 ${detectedFaces.size} 张人脸")

            // 更新UI
            withContext(Dispatchers.Main) {
                onFacesDetected(detectedFaces)
            }
        } catch (e: Exception) {
            Log.e("RealTimeBackDetection", "图像处理错误: ${e.message}", e)
            withContext(Dispatchers.Main) {
                onFacesDetected(emptyList())
            }
        } finally {
            imageProxy.close()
        }
    }
}
