package com.example.face_detect_mk

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
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
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * 图片人脸检测界面
 */
@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun ImageFaceDetection() {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    var imageUri by remember { mutableStateOf<Uri?>(null) }
    var detectedFaces by remember { mutableStateOf<List<FaceData>>(emptyList()) }
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var imageWidth by remember { mutableFloatStateOf(0f) }
    var imageHeight by remember { mutableFloatStateOf(0f) }
    var isProcessing by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    
    // ONNX Face Detector instance
    val faceDetector = remember { OnnxFaceDetector(context) }
    
    // Initialize detector
    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            if (!faceDetector.initialize()) {
                withContext(Dispatchers.Main) {
                    errorMessage = "Failed to initialize face detector"
                }
            }
        }
    }
    
    // Cleanup on dispose
    DisposableEffect(Unit) {
        onDispose {
            faceDetector.cleanup()
        }
    }
    
    val readPermissionState = when {
        android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU -> {
            rememberPermissionState(permission = android.Manifest.permission.READ_MEDIA_IMAGES)
        }
        else -> {
            rememberPermissionState(permission = android.Manifest.permission.READ_EXTERNAL_STORAGE)
        }
    }
    
    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            imageUri = it
            bitmap = getBitmapFromUri(context, it)
            bitmap?.let { bmp -> 
                imageWidth = bmp.width.toFloat()
                imageHeight = bmp.height.toFloat()
                isProcessing = true
                errorMessage = null
                
                coroutineScope.launch {
                    try {
                        val faces = withContext(Dispatchers.IO) {
                            detectFacesWithOnnx(faceDetector, bmp)
                        }
                        detectedFaces = faces
                    } catch (e: Exception) {
                        Log.e("ImageDetection", "Error detecting faces: ${e.message}", e)
                        errorMessage = "Error detecting faces: ${e.message}"
                        detectedFaces = emptyList()
                    } finally {
                        isProcessing = false
                    }
                }
            }
        }
    }
      Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Top
    ) {
        Button(
            onClick = {
                if (readPermissionState.status.isGranted) {
                    galleryLauncher.launch("image/*")
                } else {
                    readPermissionState.launchPermissionRequest()
                }
            },
            enabled = !isProcessing
        ) {
            Text(text = if (isProcessing) "处理中..." else "选择图片")
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
        // Error message
        errorMessage?.let { message ->
            Text(
                text = message,
                color = Color.Red,
                fontSize = 14.sp
            )
            Spacer(modifier = Modifier.height(8.dp))
        }
        
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(1f)
                .border(1.dp, Color.Gray),
            contentAlignment = Alignment.Center
        ) {
            imageUri?.let { uri ->
                AsyncImage(
                    model = uri,
                    contentDescription = "Selected image",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxSize()
                )
                
                // Draw face detection rectangles
                Canvas(modifier = Modifier.fillMaxSize()) {
                    if (imageWidth > 0 && imageHeight > 0) {
                        val scaleX = size.width / imageWidth
                        val scaleY = size.height / imageHeight
                        
                        detectedFaces.forEach { face ->
                            val scaledLeft = face.left * scaleX
                            val scaledTop = face.top * scaleY
                            val scaledRight = face.right * scaleX
                            val scaledBottom = face.bottom * scaleY
                            
                            drawRect(
                                color = Color(0xFF00FF00), // Green
                                topLeft = Offset(scaledLeft, scaledTop),
                                size = androidx.compose.ui.geometry.Size(scaledRight - scaledLeft, scaledBottom - scaledTop),
                                style = Stroke(width = 3f)
                            )
                              // Draw confidence text
                            val confidenceText = "%.1f%%".format(face.confidence * 100)
                            drawText(
                                text = confidenceText,
                                x = scaledLeft + 5f,
                                y = scaledTop - 5f,
                                color = Color(0xFF00FF00) // Green color
                            )
                        }
                    }
                }
            } ?: run {
                Text(text = "请选择一张图片")
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

/**
 * Convert URI to Bitmap
 */
fun getBitmapFromUri(context: Context, uri: Uri): Bitmap? {
    return OnnxFaceDetectionUtils.getBitmapFromUri(context, uri)
}

/**
 * Detect faces using ONNX model
 */
fun detectFacesWithOnnx(faceDetector: OnnxFaceDetector, bitmap: Bitmap): List<FaceData> {
    return if (faceDetector.isInitialized()) {
        faceDetector.detectFaces(bitmap)
    } else {
        Log.e("ImageDetection", "ONNX Face Detector not initialized")
        emptyList()
    }
}
