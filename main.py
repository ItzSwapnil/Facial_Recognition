"""
Enhanced Facial Recognition System - Main Entry Point
===================================================

Professional main entry point that integrates both the legacy system
and the ultra-modern 2025 SOTA face recognition system.
"""

import sys
import argparse
import logging
import asyncio
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import both systems for backward compatibility
from face_recognition_system.enhanced_main_system import EnhancedFacialRecognitionSystem
from face_recognition_system.config import load_config, get_config
from professional_main import ProfessionalFaceRecognitionSystem
import typer
from rich.console import Console

console = Console()

def main():
    """Main entry point with option to use legacy or ultra-modern system."""
    parser = argparse.ArgumentParser(
        description="Enhanced Facial Recognition System - Supports Both Legacy and Ultra-Modern",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                  # Legacy system - console interface
  python main.py --ultra-modern                   # Ultra-modern 2025 SOTA system
  python main.py --source 1                       # Legacy system with camera 1
  python main.py --ultra-modern --camera 0        # Ultra-modern with camera 0
  python main.py --add-person "John" image.jpg    # Legacy system - add person
  python main.py --capabilities                   # Show system capabilities
        """
    )
    
    # System selection
    parser.add_argument(
        "--ultra-modern",
        action="store_true",
        help="Use the ultra-modern 2025 SOTA system instead of legacy"
    )
    
    # Legacy system arguments (for backward compatibility)
    parser.add_argument(
        "--source", "-s",
        default=0,
        help="Video source: camera index (0, 1, ...), file path, or stream URL"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without video display window"
    )
    
    parser.add_argument(
        "--console",
        action="store_true",
        help="Run interactive console interface"
    )
    
    # Common arguments for both systems
    parser.add_argument(
        "--add-person",
        nargs=2,
        metavar=("NAME", "IMAGE_PATH"),
        help="Add a person to the database"
    )
    
    parser.add_argument(
        "--remove-person",
        help="Remove a person from the database"
    )
    
    parser.add_argument(
        "--list-people",
        action="store_true",
        help="List all known people"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show system statistics"
    )
    
    parser.add_argument(
        "--test-alerts",
        action="store_true",
        help="Test alert system"
    )
    
    parser.add_argument(
        "--capabilities",
        action="store_true",
        help="Show system capabilities"
    )
    
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Test all system components"
    )
    
    # Ultra-modern system specific arguments
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for ultra-modern system"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.ultra_modern:
            # Use ultra-modern system
            console.print("[bold blue]🚀 Initializing Ultra-Modern Face Recognition System (2025 SOTA)...[/bold blue]")
            
            # Convert legacy args to ultra-modern format
            ultra_args = argparse.Namespace()
            ultra_args.config = Path(args.config) if args.config else None
            ultra_args.camera = args.camera
            ultra_args.add_person = args.add_person
            ultra_args.remove_person = args.remove_person
            ultra_args.list_people = args.list_people
            ultra_args.stats = args.stats
            ultra_args.test = args.test_all
            ultra_args.capabilities = args.capabilities
            ultra_args.console = args.console or (not any([
                args.add_person, args.remove_person, args.list_people, 
                args.stats, args.test_all, args.capabilities
            ]))
            ultra_args.start = not ultra_args.console
            ultra_args.verbose = args.verbose
            ultra_args.debug = args.debug
            
            # Initialize and run ultra-modern system
            system = ProfessionalFaceRecognitionSystem(ultra_args.config)
            
            if ultra_args.debug:
                system.config.debug_mode = True
                system.config.logging.log_level = "DEBUG"
            
            # Execute appropriate command
            if ultra_args.capabilities:
                capabilities = system.get_system_capabilities()
                for category, features in capabilities.items():
                    console.print(f"\n[bold cyan]{category.title()} Capabilities:[/bold cyan]")
                    for feature, available in features.items():
                        status = "[green]✅[/green]" if available else "[red]❌[/red]"
                        console.print(f"  {status} {feature}")
                        
            elif ultra_args.test:
                console.print("\n[bold cyan]Running System Tests...[/bold cyan]")
                system.run_system_tests()
                
            elif ultra_args.add_person:
                name, image_path = ultra_args.add_person
                if system.add_person_from_image(name, Path(image_path)):
                    console.print(f"[green]✅ Added {name} successfully[/green]")
                else:
                    console.print(f"[red]❌ Failed to add {name}[/red]")
                    
            elif ultra_args.remove_person:
                if system.remove_person(ultra_args.remove_person):
                    console.print(f"[green]✅ Removed {ultra_args.remove_person} successfully[/green]")
                else:
                    console.print(f"[red]❌ Failed to remove {ultra_args.remove_person}[/red]")
                    
            elif ultra_args.list_people:
                people = system.list_people()
                if people:
                    console.print(f"\n[bold cyan]Known People ({len(people)}):[/bold cyan]")
                    for person in people:
                        console.print(f"  • {person}")
                else:
                    console.print("[yellow]No known people in database[/yellow]")
                    
            elif ultra_args.stats:
                system.print_stats()
                
            elif ultra_args.console:
                system.run_interactive_console()
                
            elif ultra_args.start:
                system.start_face_recognition(ultra_args.camera)
                
        else:
            # Use legacy system
            console.print("[bold blue]Initializing Enhanced Facial Recognition System (Legacy)...[/bold blue]")
            system = EnhancedFacialRecognitionSystem(args.config)
            
            # Handle special commands
            if args.capabilities:
                console.print("\n[bold cyan]System Capabilities:[/bold cyan]")
                capabilities = system.get_system_capabilities()
                for category, features in capabilities.items():
                    console.print(f"\n[yellow]{category.title()}:[/yellow]")
                    for feature, available in features.items():
                        status = "[green]✓[/green]" if available else "[red]✗[/red]"
                        console.print(f"  {status} {feature}")
                return
                
            if args.test_all:
                console.print("\n[bold cyan]Running System Tests...[/bold cyan]")
                system.run_system_tests()
                return
            
            # Handle different modes
            if args.console:
                # Interactive console mode
                system.run_console_interface()
                
            elif args.add_person:
                # Add person mode
                name, image_path = args.add_person
                console.print(f"Adding person: {name}")
                if system.add_person(name, image_path):
                    console.print(f"[green]✓ Successfully added {name}[/green]")
                else:
                    console.print(f"[red]✗ Failed to add {name}[/red]")
                    
            elif args.remove_person:
                # Remove person mode
                console.print(f"Removing person: {args.remove_person}")
                if system.remove_person(args.remove_person):
                    console.print(f"[green]✓ Successfully removed {args.remove_person}[/green]")
                else:
                    console.print(f"[red]✗ Failed to remove {args.remove_person}[/red]")
                    
            elif args.list_people:
                # List people mode
                people = system.recognizer.get_known_people()
                if people:
                    console.print("[bold]Known People:[/bold]")
                    for person in people:
                        console.print(f"  • {person}")
                else:
                    console.print("[yellow]No known people in database[/yellow]")
                    
            elif args.stats:
                # Stats mode
                system.print_stats()
                
            elif args.test_alerts:
                # Test alerts mode
                console.print("Testing alert system...")
                results = system.alert_system.test_notifications()
                for method, success in results.items():
                    status = "[green]✓[/green]" if success else "[red]✗[/red]"
                    console.print(f"  {status} {method}")
                    
            else:
                # Normal operation mode
                console.print("[bold green]Starting Facial Recognition System...[/bold green]")
                console.print("Press 'q' to quit, 's' to save snapshot, 'c' to clear cooldowns")
                
                # Convert source to appropriate type
                source = args.source
                if isinstance(source, str) and source.isdigit():
                    source = int(source)
                
                # Start the system
                if system.start_system(video_source=source, show_video=not args.no_display):
                    if not args.no_display:
                        # Keep running until stopped by user
                        try:
                            while system.is_running:
                                import time
                                time.sleep(0.1)
                        except KeyboardInterrupt:
                            console.print("\n[yellow]Stopping system...[/yellow]")
                            system.stop_system()
                    else:
                        # Headless mode - run until Ctrl+C
                        try:
                            console.print("Running in headless mode. Press Ctrl+C to stop.")
                            while system.is_running:
                                import time
                                time.sleep(1)
                                # Optionally print stats periodically
                                if hasattr(system, 'processed_frames') and system.processed_frames % 100 == 0:
                                    stats = system.get_system_stats()
                                    console.print(f"Processed: {stats['processed_frames']} frames, "
                                                f"FPS: {stats['processing_fps']:.1f}")
                        except KeyboardInterrupt:
                            console.print("\n[yellow]Stopping system...[/yellow]")
                            system.stop_system()
                else:
                    console.print("[red]Failed to start system[/red]")
                    sys.exit(1)
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)
    
    console.print("[green]System stopped successfully[/green]")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
