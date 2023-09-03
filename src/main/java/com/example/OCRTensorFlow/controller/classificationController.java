package com.example.OCRTensorFlow.controller;


    import com.example.OCRTensorFlow.service.TensorFlowService;
    import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

    @RestController
    @RequestMapping("/api/classify")
    public class classificationController {

        @Autowired
        private TensorFlowService tensorFlowService;

        @PostMapping("/image")
        public String classifyImage(@RequestParam("file") MultipartFile file) throws IOException {
            if (file.isEmpty()) {
                return "Please select an image file.";
            }

            byte[] imageData = file.getBytes();
            return tensorFlowService.classifyImage(imageData);
        }
    }


