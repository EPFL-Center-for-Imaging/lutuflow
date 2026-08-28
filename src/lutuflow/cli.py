import os
import argparse
from typing import Any, Callable, Dict
import questionary

import imaging_server_kit as sk

from lutuflow.omero_client._project import (
    OmeroController,
    OmeroProjectManager,
)
from lutuflow.core import NNUNET_MODELS, multikit
from lutuflow.core.cli import crop, predict, combine, track


def cli_menu(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        while True:
            clear_screen()
            out = func(*args, **kwargs)
            if out in ["Back", "back", "🔙 Back"]:
                break

    return wrapper


def clear_screen() -> None:
    """Clears the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def handle_exit(controller: OmeroController):
    print("Bye!")
    controller.quit()
    exit(0)


def handle_login() -> OmeroController:
    """Login to OMERO with username and password"""
    clear_screen()
    max_attempts = 3
    for n_attempts in range(max_attempts):
        user = questionary.text("OMERO username:").ask()
        password = questionary.password("OMERO password:").ask()

        controller = OmeroController(user, password)

        connect_status = controller.connect()
        if connect_status:
            break
        else:
            print(f"{connect_status=}")
            if n_attempts + 1 > max_attempts:
                print(f"❌ Failed to connect {max_attempts} times in a row. Exiting...")
                controller.quit()
                exit(0)

    return controller


@cli_menu
def project_menu(project: OmeroProjectManager) -> str:
    project.scanner.view.print_summary()

    project_choices = {
        "🔙 Back": "back",
        "🔁 Run detection workflow": "run_workflows",
        "⏫ Upload new scans in batch": "upload_new_scans",
        f"🐭 Select cases ({len(project.scanner.view.cases)})": "select_cases",
        f"": "",
    }

    selected_project_option = questionary.select(
        f"What to do next?",
        choices=list(project_choices.keys()),
    ).ask()

    selected_option = project_choices[selected_project_option]

    if selected_option == "run_workflows":
        clear_screen()
        if len(project.scanner.view.roi_missing) or len(project.scanner.view.pred_missing):  # type: ignore
            tumor_model = questionary.select(
                "Tumor detection model",
                choices=project.tumor_models,
            ).ask()
            if len(project.scanner.view.roi_missing) != 0:  # type: ignore
                project.batch_roi(ask_confirm=False)
            project.batch_nnunet(tumor_model, ask_confirm=False)

        input("\n✅ Press [Enter] to return to the previous menu...")

    elif selected_option == "select_cases":
        select_case_menu(project)

    elif selected_option == "upload_new_scans":
        image_dir = questionary.path(
            "Path to the parent folder containing scan directories",
            only_directories=True,
        ).ask()

        confirm = (
            input("\n✅ Press any key to confirm or [n] to cancel:").strip().lower()
        )

        if confirm == "n":
            print("❌ Cancelled.")
        else:
            for _ in project.upload_from_parent_directory(image_dir):
                pass

            input("\n✅ Press [Enter] to return to the previous menu...")

    return selected_option


@cli_menu
def select_case_menu(project: OmeroProjectManager) -> str:
    choices = ["🔙 Back"] + project.scanner.view.cases

    selected_case = questionary.select(
        "Select a case to work on", choices=choices
    ).ask()

    if selected_case in project.scanner.view.cases:
        case_menu(selected_case, project)

    return selected_case


@cli_menu
def case_menu(selected_case: str, project: OmeroProjectManager) -> str:
    print("\n" + "=" * 60)
    print(f"🐭 Selected case: {selected_case}")

    case_choices = {
        "🔙 Back": "back",
        "⏬ Download case data locally": "download_case",
    }

    selected_case_option = questionary.select(
        f"What to do next?",
        choices=list(case_choices.keys()),
    ).ask()

    selected_option = case_choices[selected_case_option]

    if selected_option == "download_case":
        out_dir = questionary.path(
            "Output path",
            default="questionary",
            only_directories=True,
        ).ask()

        project.download_case(selected_case, out_dir)

    return selected_option


@cli_menu
def interactive(controller: OmeroController) -> str:
    project_choices: Dict[str, Any] = {"🚪 Exit": None}
    for project_name, project_id in controller.projects.items():
        project_choices[f"{project_id} - {project_name}"] = (
            project_id,
            project_name,
        )

    selected_option = questionary.select(
        "Select an OMERO Project to work on",
        choices=list(project_choices.keys()),
    ).ask()

    if selected_option == "🚪 Exit":
        handle_exit(controller)

    selected_project_id, selected_project_name = project_choices[selected_option]

    project = controller.set_project(
        selected_project_id, selected_project_name, launch_scan=True
    )

    project_menu(project)

    return selected_option


def run_all_workflows(
    controller: OmeroController, project_id: int, tumor_model: str
) -> None:
    """Run all workflows on a given OMERO project"""
    for project_name, omero_project_id in controller.projects.items():
        if project_id == omero_project_id:
            break
    else:
        raise ValueError(
            f"Could not find project with ID {project_id} among available projects: {list(controller.projects.values())}"
        )

    project = controller.set_project(project_id, project_name, launch_scan=True)

    project.scanner.view.print_summary()

    if len(project.scanner.view.roi_missing) or len(project.scanner.view.pred_missing):  # type: ignore
        if len(project.scanner.view.roi_missing):  # type: ignore
            project.batch_roi(ask_confirm=False)
        project.batch_nnunet(tumor_model, ask_confirm=False)


def main():
    parser = argparse.ArgumentParser(description="LuTuFlow CLI")
    
    subparsers = parser.add_subparsers(dest="command")
    
    omero_parser = subparsers.add_parser("omero", help="Run LuTuFlow in batch and interact with OMERO projects.")
    
    omero_sub = omero_parser.add_subparsers(dest="omero_command", required=True)
    
    omero_sub.add_parser("interactive", help="Start the interactive mode.")
    
    run_parser = omero_sub.add_parser("run", help="Run the tumor detection workflow on an OMERO project.")

    run_parser.add_argument(
        "project_id",
        help="OMERO Project ID",
        type=int,
    )

    run_parser.add_argument(
        "--tumor-model",
        default="may15",
        choices=list(NNUNET_MODELS.keys()),
        help="Tumor model to use",
    )
    
    # Cropping and lungs segmentation
    crop_parser = subparsers.add_parser(
        "crop",
        help="Extract and save the lung region of interest (ROI) from a raw CT scan image file. Optionally, save the associated lung mask image.",
    )
    crop_parser.add_argument(
        "image_file",
        help="Input image file (.tif). The raw CT scan image from which to extract the lung region of interest.",
    )
    crop_parser.add_argument(
        "-o",
        "--out-dir",
        help="Directory where to save the extracted lung ROI, as a TIFF file (and, optionally, the associated lung mask image).",
    )
    crop_parser.add_argument(
        "--image-only",
        action="store_false",
        help="Use this flag to only save the ROI image, and not the corresponding lungs mask image.",
    )
    
    # Tumor segmentation
    predict_parser = subparsers.add_parser(
        "predict",
        help="Segment tumors in the lung region of interest (ROI) of mice CT scans.",
    )
    predict_parser.add_argument(
        "image_file",
        help="Input image file (.tif). The lung region of interest (ROI).",
    )
    predict_parser.add_argument(
        "-o",
        "--out-dir",
        help="Directory where to save the predicted tumor mask, as a TIFF file.",
    )
    predict_parser.add_argument(
        "-m",
        "--model",
        default="may25",
        choices=list(NNUNET_MODELS.keys()),
        help="Model to use for the tumor detection.",
    )

    # Combining images into time series
    combine_parser = subparsers.add_parser(
        "combine",
        help="Combine multiple image files (lung ROIs or tumor masks) into a single 4D time series TIFF file (TZYX).",
    )
    combine_parser.add_argument(
        "files",
        nargs="+",
        help="Files to process. The files should be orderd from the oldest to the most recent scan (scan0, scan1, scan2, etc.).",
    )
    combine_parser.add_argument(
        "-o",
        "--out-dir",
        help="Directory where to save the output (4D TIFF file).",
    )

    # Tracking tumors
    track_parser = subparsers.add_parser(
        "track",
        help="Track individual tumors in scan time series.",
    )
    track_parser.add_argument(
        "labels_file",
        help="Labels file (.tif). It should be a 4D (TZYX) time series of tumor masks.",
    )
    track_parser.add_argument(
        "image_file",
        help="Input image (.tif). It should be a 4D (TZYX) time series of lung ROIs.",
    )
    track_parser.add_argument(
        "-o",
        "--out-dir",
        help="Directory where to save the output (4D TIFF of tracked tumors).",
    )
    track_parser.add_argument(
        "--max-dist",
        type=int,
        default=30,
        help="Maximum search distance in pixels. Only tumors that move by less than this cutoff between two consecutive scans are considered potiential matches.",
    )
    track_parser.add_argument(
        "--dist-ratio",
        type=float,
        default=0.9,
        help="[0-1] Relative importance given to the distance between tumors and their similarity in size for the tracking algorithm. Decrease this value to favour matches based on volume similarity.",
    )
    track_parser.add_argument(
        "--vol-diff",
        type=float,
        default=1.0,
        help="Maximum relative volume change of an tumor between two consecutive scans. 1.0 constraints tumors to at most double in size, or shrink by half of their volume between two consecutive scans.",
    )

    # Serve
    serve_parser = subparsers.add_parser(
        "serve",
        help="Create a web server for LuTuFlow functions.",
    )
    serve_parser.add_argument(
        "--host", default="0.0.0.0", help="Server address (default: 0.0.0.0)."
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)."
    )    

    args = parser.parse_args()
    
    if args.command == "omero":
        if args.omero_command == "interactive":
            controller = handle_login()
            interactive(controller)

        elif args.omero_command == "run":
            controller = handle_login()
            run_all_workflows(controller, args.project_id, args.tumor_model)

    elif args.command == "crop":
        crop(args.image_file, args.out_dir, args.image_only)
        
    elif args.command == "predict":
        predict(args.image_file, args.out_dir, args.model)
        
    elif args.command == "combine":
        combine(*args.files, out_dir=args.out_dir)
        
    elif args.command == "track":
        track(
            labels_file=args.labels_file,
            image_file=args.image_file,
            out_dir=args.out_dir,
            max_dist_px=args.max_dist,
            max_volume_diff_rel=args.vol_diff,
            dist_weight_ratio=args.dist_ratio,
        )
        
    elif args.command == "serve":
        sk.serve(multikit, host=args.host, port=args.port)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
