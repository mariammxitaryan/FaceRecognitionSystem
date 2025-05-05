import argparse
import json
import sys
from deepface import DeepFace


# ────────────────────────────────────────────────────────────────────────────────
# Function: recognize_face
# Purpose : Given a query image, search a directory of faces and return
#           the closest matches (sorted by embedding distance).
#
# Arguments:
#   • img_path         Path to the image you want to recognize.
#   • db_path          Directory containing one face image per person.
#   • model_name       Embedding model to use (VGG‑Face, Facenet, ArcFace…).
#   • detector_backend Face detector to use (opencv, mtcnn, dlib, retinaface).
#   • enforce_detection If False, won’t error on “no face found”.
#
# Returns : A list of dicts, each with “identity” and distance scores.
# ────────────────────────────────────────────────────────────────────────────────
def recognize_face(
    img_path: str,
    db_path: str,
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    enforce_detection: bool = True
) -> list:
    try:
        # Call DeepFace.find() to compare the query face against your DB folder
        result_df = DeepFace.find(
            img_path=img_path,
            db_path=db_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection
        )
        # Convert the resulting DataFrame of matches into a list of dicts
        return result_df.to_dict(orient="records")
    except Exception as e:
        # Print any errors to stderr and return an empty list
        print(f"[ERROR] Recognition failed: {e}", file=sys.stderr)
        return []


# ────────────────────────────────────────────────────────────────────────────────
# Function: analyze_face
# Purpose : Detect a face in an image and estimate attributes:
#           age, gender, race composition, emotion scores.
#
# Arguments:
#   • img_path         Path to the image to analyze.
#   • actions          List of analyses to run (["age","gender","race","emotion"]).
#   • model_name       Embedding model (Facenet, VGG‑Face, ArcFace…).
#   • detector_backend Face detector (mtcnn, opencv, dlib, retinaface).
#   • enforce_detection If False, won’t error if no face found.
#   • output_json      If provided, write a JSON report to this filepath.
#
# Returns : A dict containing requested attributes and their values.
# ────────────────────────────────────────────────────────────────────────────────
def analyze_face(
    img_path: str,
    actions: list = None,
    model_name: str = "Facenet",
    detector_backend: str = "mtcnn",
    enforce_detection: bool = True,
    output_json: str = None
) -> dict:
    # Default to all four analyses if none specified
    if actions is None:
        actions = ["age", "gender", "race", "emotion"]

    try:
        # Run DeepFace.analyze() to get attribute predictions
        result = DeepFace.analyze(
            img_path=img_path,
            actions=actions,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection
        )
        # Optionally dump the result dict to a JSON file
        if output_json:
            with open(output_json, "w", encoding="utf-8") as fp:
                json.dump(result, fp, indent=4, ensure_ascii=False)
        return result
    except Exception as e:
        # Print any errors to stderr and return an empty dict
        print(f"[ERROR] Analysis failed: {e}", file=sys.stderr)
        return {}


# ────────────────────────────────────────────────────────────────────────────────
# Function: verify_faces
# Purpose : Compare two face images and decide if they show the same person.
#
# Arguments:
#   • img1             First image filepath.
#   • img2             Second image filepath.
#   • model_name       Embedding model (ArcFace, Facenet, VGG‑Face…).
#   • distance_metric  How to compute similarity (cosine, euclidean…).
#   • detector_backend Face detector (dlib, mtcnn, opencv, retinaface).
#   • enforce_detection If False, won’t error if no face found.
#
# Returns : A dict with keys:
#           – verified (bool)
#           – distance (float)
#           – threshold (float)
# ────────────────────────────────────────────────────────────────────────────────
def verify_faces(
    img1: str,
    img2: str,
    model_name: str = "ArcFace",
    distance_metric: str = "cosine",
    detector_backend: str = "dlib",
    enforce_detection: bool = True
) -> dict:
    try:
        # Call DeepFace.verify() to compare embeddings of two images
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name=model_name,
            distance_metric=distance_metric,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection
        )
        # Return only the keys you care about
        return {
            "verified": result.get("verified", False),
            "distance": result.get("distance", None),
            "threshold": result.get("threshold", None)
        }
    except Exception as e:
        # Print any errors to stderr and return an empty dict
        print(f"[ERROR] Verification failed: {e}", file=sys.stderr)
        return {}


def main():
    # Initialize the top-level CLI parser
    parser = argparse.ArgumentParser(
        description="Face Recognition / Analysis CLI using DeepFace"
    )
    # Create sub-command parsers: recognize, analyze, verify
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ─── recognize sub-command ───────────────────────────────────────────────────
    p_rec = subparsers.add_parser(
        "recognize", help="Find a face in a DB folder"
    )
    p_rec.add_argument("--img", required=True, help="Path to input image")
    p_rec.add_argument("--db", required=True, help="Path to face database folder")
    p_rec.add_argument("--model", default="VGG-Face", help="Embedding model")
    p_rec.add_argument("--backend", default="opencv", help="Detector backend")
    p_rec.add_argument(
        "--no-enforce",
        action="store_false",
        dest="enforce_detection",
        help="Skip enforcing face detection"
    )

    # ─── analyze sub-command ────────────────────────────────────────────────────
    p_an = subparsers.add_parser(
        "analyze", help="Analyze face attributes"
    )
    p_an.add_argument("--img", required=True, help="Path to input image")
    p_an.add_argument(
        "--actions",
        nargs="+",
        default=["age", "gender", "race", "emotion"],
        help="Attributes to analyze"
    )
    p_an.add_argument("--model", default="Facenet", help="Embedding model")
    p_an.add_argument("--backend", default="mtcnn", help="Detector backend")
    p_an.add_argument(
        "--no-enforce",
        action="store_false",
        dest="enforce_detection",
        help="Skip enforcing face detection"
    )
    p_an.add_argument("--out", help="Path to dump JSON output")

    # ─── verify sub-command ─────────────────────────────────────────────────────
    p_ver = subparsers.add_parser(
        "verify", help="Verify two face images"
    )
    p_ver.add_argument("--img1", required=True, help="First face image")
    p_ver.add_argument("--img2", required=True, help="Second face image")
    p_ver.add_argument("--model", default="ArcFace", help="Embedding model")
    p_ver.add_argument("--metric", default="cosine", help="Distance metric")
    p_ver.add_argument("--backend", default="dlib", help="Detector backend")
    p_ver.add_argument(
        "--no-enforce",
        action="store_false",
        dest="enforce_detection",
        help="Skip enforcing face detection"
    )

    # Parse arguments and dispatch to the chosen function
    args = parser.parse_args()

    if args.command == "recognize":
        hits = recognize_face(
            img_path=args.img,
            db_path=args.db,
            model_name=args.model,
            detector_backend=args.backend,
            enforce_detection=args.enforce_detection
        )
        print(json.dumps(hits, indent=2, ensure_ascii=False))

    elif args.command == "analyze":
        info = analyze_face(
            img_path=args.img,
            actions=args.actions,
            model_name=args.model,
            detector_backend=args.backend,
            enforce_detection=args.enforce_detection,
            output_json=args.out
        )
        print(json.dumps(info, indent=2, ensure_ascii=False))

    elif args.command == "verify":
        verdict = verify_faces(
            img1=args.img1,
            img2=args.img2,
            model_name=args.model,
            distance_metric=args.metric,
            detector_backend=args.backend,
            enforce_detection=args.enforce_detection
        )
        print(json.dumps(verdict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
