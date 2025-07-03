"""
Setup Script for Latest Face Recognition System
==============================================

This script helps you set up the face recognition system by:
1. Adding your face to the known faces database
2. Testing the camera setup
3. Configuring notifications
"""

import cv2
import os
import json
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from latest_face_recognition import LatestFaceRecognitionSystem

def setup_directories():
    """Create necessary directories"""
    directories = [
        "data/known_faces",
        "data/models",
        "detections",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")

def capture_your_face(console: Console):
    """Capture your face for recognition"""
    console.print("[bold blue]📸 Face Capture Setup[/bold blue]")
    console.print("We'll now capture your face for recognition.")
    console.print("Look directly at the camera and press SPACE when ready, ESC to cancel.")
    
    # Get your name
    your_name = Prompt.ask("What's your name?", default="User")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        console.print("[red]❌ Could not open camera[/red]")
        return False
    
    console.print("[green]Camera opened. Position yourself in front of the camera.[/green]")
    console.print("[yellow]Press SPACE to capture, ESC to cancel[/yellow]")
    
    captured = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show preview
        cv2.putText(frame, "Press SPACE to capture, ESC to cancel", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Capturing for: {your_name}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow('Face Capture - Position yourself and press SPACE', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space to capture
            # Save the image
            image_path = f"data/known_faces/{your_name}.jpg"
            cv2.imwrite(image_path, frame)
            console.print(f"[green]✅ Face captured and saved as {image_path}[/green]")
            captured = True
            break
        elif key == 27:  # ESC to cancel
            console.print("[yellow]❌ Capture cancelled[/yellow]")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if captured:
        # Add to face recognition system
        system = LatestFaceRecognitionSystem()
        success = system.add_known_person(your_name, image_path)
        if success:
            console.print(f"[green]✅ {your_name} added to face recognition database[/green]")
            return True
        else:
            console.print("[red]❌ Failed to add face to database[/red]")
    
    return False

def test_camera(console: Console):
    """Test camera functionality"""
    console.print("[bold blue]🎥 Camera Test[/bold blue]")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        console.print("[red]❌ Could not open camera[/red]")
        return False
    
    console.print("[green]Camera opened successfully![/green]")
    console.print("[yellow]Press any key to close the test window[/yellow]")
    
    ret, frame = cap.read()
    if ret:
        cv2.putText(frame, "Camera Test - Press any key to close", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Camera Test', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        console.print("[green]✅ Camera test completed[/green]")
    
    cap.release()
    return True

def configure_system(console: Console):
    """Configure system settings"""
    console.print("[bold blue]⚙️ System Configuration[/bold blue]")
    
    config = {
        "detection_threshold": 0.7,
        "recognition_threshold": 0.6,
        "notification_cooldown": 30,
        "save_detections": True,
        "detection_save_path": "detections/",
        "known_faces_path": "data/known_faces/",
        "models_path": "data/models/",
        "camera_source": 0,
        "frame_skip": 2,
        "notification_sound": True,
        "notification_popup": True,
        "notification_log": True
    }
    
    # Ask for configuration preferences
    if Confirm.ask("Do you want to enable sound notifications?", default=True):
        config["notification_sound"] = True
    else:
        config["notification_sound"] = False
    
    if Confirm.ask("Do you want to enable popup notifications?", default=True):
        config["notification_popup"] = True
    else:
        config["notification_popup"] = False
    
    # Save configuration
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    console.print("[green]✅ Configuration saved[/green]")

def main():
    """Main setup function"""
    console = Console()
    
    console.print("[bold green]🚀 Face Recognition System Setup[/bold green]")
    console.print("Welcome! Let's set up your face recognition system.")
    
    # Step 1: Create directories
    console.print("\n[bold]Step 1: Setting up directories[/bold]")
    setup_directories()
    
    # Step 2: Test camera
    console.print("\n[bold]Step 2: Testing camera[/bold]")
    if not test_camera(console):
        console.print("[red]❌ Camera test failed. Please check your camera connection.[/red]")
        return
    
    # Step 3: Configure system
    console.print("\n[bold]Step 3: Configuring system[/bold]")
    configure_system(console)
    
    # Step 4: Capture your face
    console.print("\n[bold]Step 4: Adding your face[/bold]")
    if capture_your_face(console):
        console.print("\n[bold green]🎉 Setup Complete![/bold green]")
        console.print("Your face recognition system is ready to use!")
        console.print("\nTo start the system, run:")
        console.print("[bold cyan]uv run python latest_face_recognition.py[/bold cyan]")
        
        if Confirm.ask("\nWould you like to start the system now?", default=True):
            console.print("\n[bold blue]Starting Face Recognition System...[/bold blue]")
            system = LatestFaceRecognitionSystem()
            system.start_detection()
    else:
        console.print("[yellow]⚠️ Setup incomplete. You can run this setup again later.[/yellow]")

if __name__ == "__main__":
    main()
