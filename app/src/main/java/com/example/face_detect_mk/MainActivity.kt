package com.example.face_detect_mk

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
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
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.example.face_detect_mk.ui.theme.Face_detect_MKTheme
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            Face_detect_MKTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    FaceDetectionScreen(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun FaceDetectionScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var imageUri by remember { mutableStateOf<Uri?>(null) }
    var detectedFaces by remember { mutableStateOf<List<Face>>(emptyList()) }
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var imageWidth by remember { mutableFloatStateOf(0f) }
    var imageHeight by remember { mutableFloatStateOf(0f) }
    
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
                detectFaces(context, bmp) { faces ->
                    detectedFaces = faces
                }
            }
        }
    }
    
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Top
    ) {
        Text(
            text = "人脸检测", 
            fontSize = 24.sp,
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.padding(bottom = 16.dp)
        )
        
        Spacer(modifier = Modifier.height(16.dp))
        
        Button(
            onClick = {
                if (readPermissionState.status.isGranted) {
                    galleryLauncher.launch("image/*")
                } else {
                    readPermissionState.launchPermissionRequest()
                }
            }
        ) {
            Text(text = "选择图片")
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
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
                
                // Draw face rectangles over the image
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val scaleX = size.width / imageWidth
                    val scaleY = size.height / imageHeight
                    
                    detectedFaces.forEach { face ->
                        val rect = face.boundingBox
                        val scaledLeft = rect.left * scaleX
                        val scaledTop = rect.top * scaleY
                        val scaledRight = rect.right * scaleX
                        val scaledBottom = rect.bottom * scaleY
                        
                        drawRect(
                            color = Color.Green,
                            topLeft = Offset(scaledLeft, scaledTop),
                            size = Size(scaledRight - scaledLeft, scaledBottom - scaledTop),
                            style = Stroke(width = 3f)
                        )
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

fun getBitmapFromUri(context: Context, uri: Uri): Bitmap? {
    return try {
        val inputStream = context.contentResolver.openInputStream(uri)
        BitmapFactory.decodeStream(inputStream)
    } catch (e: Exception) {
        Log.e("FaceDetection", "Error loading bitmap: ${e.message}")
        null
    }
}

fun detectFaces(context: Context, bitmap: Bitmap, onComplete: (List<Face>) -> Unit) {
    // High-accuracy landmark detection and face classification
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