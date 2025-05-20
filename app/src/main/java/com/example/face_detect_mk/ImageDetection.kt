package com.example.face_detect_mk

import android.graphics.Bitmap
import android.net.Uri
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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.mlkit.vision.face.Face

/**
 * 图片人脸检测界面
 */
@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun ImageFaceDetection() {
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
                
                // 在图片上绘制人脸识别矩形
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
                            size = androidx.compose.ui.geometry.Size(scaledRight - scaledLeft, scaledBottom - scaledTop),
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
