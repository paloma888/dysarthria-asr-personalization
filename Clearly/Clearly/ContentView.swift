//
//  ContentView.swift
//  Clearly
//
//  Created by Paloma Pichardo on 7/19/26.
//

import SwiftUI
import AVFoundation

enum AppState: Equatable {
    case idle
    case recording
    case processing
    case done(String)
    case error(String)
}

struct TranscribeResponse: Codable {
    let text: String
    let latency_ms: Double
}

struct ContentView: View {
    @State private var appState: AppState = .idle
    @State private var audioRecorder: AVAudioRecorder?
    @State private var speechSynthesizer = AVSpeechSynthesizer()
    
    var body: some View {
        VStack {
            Button(buttonLabel(for: appState), action: {switch appState {
            case .idle:
                if AVAudioApplication.shared.recordPermission == .granted {
                    appState = .recording
                    startRecording()
                } else {
                    appState = .error("Microphone access denied")
                }
            case .recording:
                stopRecording()
                appState = .processing
                Task {
                    do {
                        let text = try await transcribeAudio(fileURL: getRecordingURL())
                        appState = .done(text)
                    } catch {
                        appState = .error("Could not reach server")
                    }
                }
            case .processing:
                break
            case .done(_):
                appState = .idle
            case .error(_):
                appState = .idle
            }})
            .font(.title)
            .frame(width: 250)
            .padding()
            .background(Color .blue)
            .foregroundColor(.white)
            .cornerRadius(15)
            .padding()
            
            Text(statusMessage(for: appState))
        }
        .padding()
        .onAppear {
            requestMicPermission()
            configureAudioSession()
        }
        .onChange(of: appState) { oldState, newState in
            if case .done(let text) = newState {
                speak(text)
            }
        }
    }
    
    func statusMessage(for state: AppState) -> String {
        switch state{
        case .idle:
            return "Press to record"
        case .recording:
            return "Listening..."
        case .processing:
            return "Processing..."
        case .done(let text):
            return "Result: \(text)"
        case .error(let msg):
            return "Error: \(msg)"
            
        }
    }
    
    func buttonLabel(for state: AppState) -> String {
        switch state {
        case .idle:
            return "Start Recording"
        case .recording:
            return "Stop Recording"
        case .processing:
            return "Processing..."
        case .done(_):
            return "Record Again"
        case .error(_):
            return "Try Again"
        }
    }
    
    func requestMicPermission() {
        AVAudioApplication.requestRecordPermission { granted in
            if !granted {
                appState = .error("Microphone access denied")
            }
        }
    }
    
    func configureAudioSession() {
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        try? session.setActive(true)
    }
    
    func getRecordingURL() -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        return tempDir.appendingPathComponent("recording.wav")
    }
    
    let recorderSettings: [String: Any] = [
        AVFormatIDKey: Int(kAudioFormatLinearPCM),
        AVSampleRateKey: 16000,
        AVNumberOfChannelsKey: 1,
        AVLinearPCMBitDepthKey: 16,
        AVLinearPCMIsFloatKey: false
    ]
    
    func startRecording() {
        let url = getRecordingURL()
        do {
            audioRecorder = try AVAudioRecorder(url: url, settings: recorderSettings)
            audioRecorder?.record()
        } catch {
            appState = .error("Could not start recording")
        }
    }
    
    
    func stopRecording() {
        audioRecorder?.stop()
        audioRecorder = nil
        let url = getRecordingURL()
        print("File exists:", FileManager.default.fileExists(atPath: url.path))
    }
    
    func transcribeAudio(fileURL: URL) async throws -> String {
        let serverURL = URL(string: "http://3.144.166.183:8000/transcribe")!
        var request = URLRequest(url: serverURL)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        let audioData = try Data(contentsOf: fileURL)

        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"recording.wav\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
        body.append(audioData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

        request.httpBody = body
        
        let (data, _) = try await URLSession.shared.data(for: request)
        let decoded = try JSONDecoder().decode(TranscribeResponse.self, from: data)
        return decoded.text
        
    }
    
    func speak(_ text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        speechSynthesizer.speak(utterance)
    }
    
}

#Preview {
    ContentView()
}
