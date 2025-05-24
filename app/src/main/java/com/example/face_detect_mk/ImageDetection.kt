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
    var faceEmotions by remember { mutableStateOf<Map<Int, EmotionData>>(emptyMap()) }
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var imageWidth by remember { mutableFloatStateOf(0f) }
    var imageHeight by remember { mutableFloatStateOf(0f) }
    var isProcessing by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    
    // ONNX Face Detector instance
    val faceDetector = remember { OnnxFaceDetector(context) }
    
    // ONNX Emotion Detector instance
    val emotionDetector = remember { OnnxEmotionDetector(context) }
      // Initialize detectors
    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            val faceInitialized = faceDetector.initialize()
            val emotionInitialized = emotionDetector.initialize()
            
            if (!faceInitialized || !emotionInitialized) {
                withContext(Dispatchers.Main) {
                    errorMessage = "Failed to initialize detectors (Face: $faceInitialized, Emotion: $emotionInitialized)"
                }
            }
        }
    }
    
    // Cleanup on dispose
    DisposableEffect(Unit) {
        onDispose {
            faceDetector.cleanup()
            emotionDetector.cleanup()
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
                        
                        // Detect emotions for each face
                        if (faces.isNotEmpty()) {
                            val emotions = withContext(Dispatchers.IO) {
                                detectEmotionsForFaces(emotionDetector, bmp, faces)
                            }
                            faceEmotions = emotions
                        } else {
                            faceEmotions = emptyMap()
                        }
                    } catch (e: Exception) {
                        Log.e("ImageDetection", "Error detecting faces/emotions: ${e.message}", e)
                        errorMessage = "Error detecting faces/emotions: ${e.message}"
                        detectedFaces = emptyList()
                        faceEmotions = emptyMap()
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
                        
                        Log.d("ImageDetection", "Canvas size: ${size.width}x${size.height}, " +
                              "Image size: ${imageWidth}x${imageHeight}, " +
                              "Scale: ${scaleX}x${scaleY}")
                          detectedFaces.forEachIndexed { index, face ->
                            val scaledLeft = face.left * scaleX
                            val scaledTop = face.top * scaleY
                            val scaledRight = face.right * scaleX
                            val scaledBottom = face.bottom * scaleY
                            
                            Log.d("ImageDetection", "Face $index: original=(${face.left}, ${face.top}, ${face.right}, ${face.bottom}), " +
                                  "scaled=(${scaledLeft}, ${scaledTop}, ${scaledRight}, ${scaledBottom})")
                            
                            drawRect(
                                color = Color(0xFF00FF00), // Green
                                topLeft = Offset(scaledLeft, scaledTop),
                                size = androidx.compose.ui.geometry.Size(scaledRight - scaledLeft, scaledBottom - scaledTop),
                                style = Stroke(width = 3f)
                            )
                              // Draw confidence and emotion text
                            val confidenceText = "%.1f%%".format(face.confidence * 100)
                            val emotion = faceEmotions[index]
                            val emotionText = emotion?.let { 
                                "${it.emotion} (%.1f%%)".format(it.confidence * 100) 
                            } ?: "Analyzing..."
                            
                            val displayText = "$confidenceText | $emotionText"
                            
                            drawText(
                                text = displayText,
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
        
        // Display emotion details
        if (faceEmotions.isNotEmpty()) {
            Spacer(modifier = Modifier.height(8.dp))
            faceEmotions.forEach { (index, emotion) ->
                Text(
                    text = "人脸 ${index + 1}: ${emotion.emotion} (${(emotion.confidence * 100).toInt()}%)",
                    fontSize = 14.sp,
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
        }
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

/**
 * Detect emotions for detected faces
 */
fun detectEmotionsForFaces(
    emotionDetector: OnnxEmotionDetector, 
    bitmap: Bitmap, 
    faces: List<FaceData>
): Map<Int, EmotionData> {
    if (!emotionDetector.isInitialized()) {
        Log.e("ImageDetection", "ONNX Emotion Detector not initialized")
        return emptyMap()
    }
    
    val emotions = mutableMapOf<Int, EmotionData>()
    
    faces.forEachIndexed { index, face ->
        try {
            // Convert FaceData to Rect for emotion detection
            val faceRect = android.graphics.Rect(
                face.left.toInt(),
                face.top.toInt(), 
                face.right.toInt(),
                face.bottom.toInt()
            )
            
            Log.d("ImageDetection", "Detecting emotion for face $index at $faceRect")
            
            val emotion = emotionDetector.detectEmotionFromFaceRegion(bitmap, faceRect)
            if (emotion != null) {
                emotions[index] = emotion
                Log.d("ImageDetection", "Face $index emotion: ${emotion.emotion} (${(emotion.confidence * 100).toInt()}%)")
            } else {
                Log.w("ImageDetection", "Failed to detect emotion for face $index")
            }
        } catch (e: Exception) {
            Log.e("ImageDetection", "Error detecting emotion for face $index", e)
        }
    }
    
    return emotions
}
