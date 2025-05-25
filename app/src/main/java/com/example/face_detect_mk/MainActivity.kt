package com.example.face_detect_mk

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.face_detect_mk.ui.theme.Face_detect_MKTheme

/**
 * 主活动类
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            Face_detect_MKTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    FaceDetectionApp(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}

/**
 * 主应用界面，包含标签页
 */
@Composable
fun FaceDetectionApp(modifier: Modifier = Modifier) {
    var selectedTabIndex by remember { mutableIntStateOf(0) }
    
    Column(
        modifier = modifier.fillMaxSize()
    ) {
        Text(
            text = "Emotion Detection", 
            fontSize = 24.sp,
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.padding(16.dp)
        )
        
        TabRow(selectedTabIndex = selectedTabIndex) {
            Tab(
                selected = selectedTabIndex == 0,
                onClick = { selectedTabIndex = 0 },
                text = { Text("Image") }
            )
            Tab(
                selected = selectedTabIndex == 1,
                onClick = { selectedTabIndex = 1 },
                text = { Text("Real-time") }
            )
        }
        
        when (selectedTabIndex) {
            0 -> ImageFaceDetection()
            1 -> RealTimeFaceDetection()
        }
    }
}