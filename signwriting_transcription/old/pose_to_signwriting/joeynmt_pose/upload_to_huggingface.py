import os
import subprocess
from huggingface_hub import HfApi
from signwriting_transcription.pose_to_signwriting.joeynmt_pose.upload_to_sheets import upload_line


def update_model_info(info, additional_name_file=""):
    file_path = info["model_dir"]
    full_path = os.path.abspath(file_path)

    commit_id = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
    commit_description = (subprocess.check_output(["git", "show", "--format=%B", "-s", commit_id]).strip()
                          .decode("utf-8"))

    api = HfApi()
    api.upload_file(
        path_or_fileobj=full_path,
        path_in_repo=f'{additional_name_file}{commit_id}.ckpt',
        repo_id="ohadlanger/signwriting_transcription",
        repo_type="space",
        token=info["token"],
    )

    print("Model uploaded!")

    new_line = [f'f{commit_id}.ckpt', info["SymbolScore"], info["BleuScore"],
                info["ChrfScore"], info["ClipScore"], info["SymbolScore_dev"], info["BleuScore_dev"],
                info["ChrfScore_dev"], info["ClipScore_dev"], commit_description]
    upload_line(new_line)
