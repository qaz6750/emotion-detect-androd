package com.example.face_detect_mk

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * 实时相机人脸检测界面
 */
@androidx.annotation.OptIn(ExperimentalGetImage::class)
@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun CameraFaceDetection() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val coroutineScope = rememberCoroutineScope()
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    // 存储已处理的图片
    var processedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    // 人脸检测状态
    var faceCount by remember { mutableStateOf(0) }
    // 记录相机绑定状态
    var cameraInitialized by remember { mutableStateOf(false) }
    // 错误状态
    var hasError by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf("") }

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
            // 更改为 FILL_CENTER 以填满视图且保持纵横比
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    // 如果权限已授予，设置相机
    LaunchedEffect(cameraPermissionState.status.isGranted) {
        if (cameraPermissionState.status.isGranted) {
            try {
                // 创建人脸检测分析器
                val faceDetector = FaceDetection.getClient(
                    FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                        .setMinFaceSize(0.2f)
                        .build()
                )

                // Corrected line:
                val cameraProvider = cameraProviderFuture.get(2, TimeUnit.SECONDS)

                // 预览用例
                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }
                // 图像分析用例 - 使用更符合手机屏幕比例的分辨率
                val imageAnalysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(720, 1280)) // 更改为 16:9 的分辨率
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

                imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    analyzeImage(imageProxy, faceDetector, coroutineScope) { bitmap, faces ->
                        processedBitmap = bitmap
                        faceCount = faces
                    }
                }

                // 解绑之前绑定的用例
                cameraProvider.unbindAll()

                // 将用例绑定到相机
                val camera = cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_FRONT_CAMERA,  // 使用前置摄像头
                    preview,
                    imageAnalysis
                )

                cameraInitialized = true
                hasError = false
                errorMessage = ""

            } catch (e: Exception) {
                Log.e("CameraFaceDetection", "Camera setup failed: ${e.message}", e)
                hasError = true
                errorMessage = "相机初始化失败: ${e.message}"
            }
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
        } else if (hasError) {
            Text(
                text = errorMessage,
                color = Color.Red,
                modifier = Modifier.padding(16.dp)
            )
        } else {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(0.7f) // 调整相机预览大小，从1.0调整为0.7
                    .padding(horizontal = 8.dp) // 添加水平内边距
                    .clip(androidx.compose.foundation.shape.RoundedCornerShape(16.dp)) // 添加圆角
                //contentAlignment = Alignment.Center
            ) {
                // 选择性显示：如果有处理后的图像，则显示处理后的图像；否则显示原始预览
                if (processedBitmap != null) {
                    // 显示处理后的图像
                    Image(
                        bitmap = processedBitmap!!.asImageBitmap(),
                        contentDescription = "Processed image with face detection",
                        contentScale = ContentScale.Crop, // 使用导入的ContentScale
                        modifier = Modifier.fillMaxSize()
                    )
                } else {
                    // 显示相机预览
                    AndroidView(
                        factory = { previewView },
                        modifier = Modifier.fillMaxSize()
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = "检测到 $faceCount 张人脸",
                fontSize = 18.sp,
                color = MaterialTheme.colorScheme.secondary
            )
        }
    }
}

// 分析图像并检测人脸
@ExperimentalGetImage
private fun analyzeImage(
    imageProxy: ImageProxy,
    faceDetector: com.google.mlkit.vision.face.FaceDetector,
    coroutineScope: kotlinx.coroutines.CoroutineScope,
    onProcessed: (Bitmap?, Int) -> Unit
) {
    val mediaImage = imageProxy.image
    if (mediaImage == null) {
        imageProxy.close()
        return
    }

    coroutineScope.launch(Dispatchers.IO) {
        try {
            // 保存旋转角度信息，前置相机需要特殊处理
            val rotation = imageProxy.imageInfo.rotationDegrees

            // 转换为ML Kit输入图像 - 我们知道这是前置相机
            val inputImage = InputImage.fromMediaImage(
                mediaImage,
                rotation
            )

            // 获取位图以便绘制
            val bitmap = imageProxyToBitmap(imageProxy)

            // 检测人脸
            var detectedFaceCount = 0
            var resultBitmap: Bitmap? = null

            try {
                // 处理面部识别
                val faces = faceDetector.process(inputImage).await()
                detectedFaceCount = faces.size

                // 如果检测到人脸，在位图上绘制标记
                if (bitmap != null) {
                    // 即使没有检测到人脸，也返回原始位图
                    resultBitmap = if (faces.isNotEmpty()) {
                        // 绘制人脸边框 - 传递前置相机标志
                        drawFacesOnBitmap(bitmap, faces)
                    } else {
                        bitmap
                    }
                }
            } catch (e: Exception) {
                Log.e("FaceDetection", "Error detecting faces: ${e.message}", e)
                resultBitmap = bitmap
            }

            // 更新UI
            withContext(Dispatchers.Main) {
                onProcessed(resultBitmap, detectedFaceCount)
            }

        } catch (e: Exception) {
            Log.e("ImageAnalysis", "Error processing image: ${e.message}", e)
            withContext(Dispatchers.Main) {
                onProcessed(null, 0)
            }
        } finally {
            imageProxy.close()
        }
    }
}

// 将ImageProxy转换为Bitmap
@ExperimentalGetImage
private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
    val image = imageProxy.image ?: return null

    try {
        // YUV_420_888 转换为 NV21
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

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 80, out)
        val imageBytes = out.toByteArray()

        val rotation = imageProxy.imageInfo.rotationDegrees
        var bitmap = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        // 根据旋转角度旋转图像
        if (rotation != 0 && bitmap != null) {
            val matrix = android.graphics.Matrix()
            matrix.postRotate(rotation.toFloat())

            // 前置相机需要水平翻转
            matrix.postScale(-1f, 1f)

            // 创建旋转后的位图
            bitmap = android.graphics.Bitmap.createBitmap(
                bitmap, 0, 0,
                bitmap.width, bitmap.height,
                matrix, true
            )
        }

        return bitmap
    } catch (e: Exception) {
        Log.e("BitmapConversion", "Error converting image to bitmap: ${e.message}", e)
        return null
    }
}

// 在位图上绘制人脸边框
private fun drawFacesOnBitmap(bitmap: Bitmap, faces: List<Face>): Bitmap {
    val resultBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(resultBitmap)
    val paint = Paint().apply {
        color = android.graphics.Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }

    val bitmapWidth = bitmap.width

    // 绘制人脸边框，需要考虑图像已被水平翻转的情况
    faces.forEach { face ->
        val bounds = face.boundingBox

        // 对边界框坐标进行转换，处理水平翻转问题
        val left = bitmapWidth - bounds.right  // 翻转左右坐标
        val right = bitmapWidth - bounds.left

        // 使用转换后的坐标绘制边界框
        canvas.drawRect(
            left.toFloat(), bounds.top.toFloat(), right.toFloat(),
            bounds.bottom.toFloat(), paint
        )

        Log.d(
            "FaceDrawing",
            "Drawing face at $left,${bounds.top} - $right,${bounds.bottom} (original: ${bounds.left},${bounds.top} - ${bounds.right},${bounds.bottom})"
        )
    }

    return resultBitmap
}
