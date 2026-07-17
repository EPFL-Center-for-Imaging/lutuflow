import re

from mousetumorpy import LungsPredictor, TumorPredictor

from depalma_napari_omero.omero_client._client import OmeroClient


def find_image_tag(img_tags) -> list:
    r = re.compile("(I|i)mage(s?)")
    image_tag = list(filter(r.match, img_tags))
    if len(image_tag) == 0:
        return []
    return image_tag


def _compute_roi(
    model,
    image_name: str,
    image_id: int,
    dataset_id: int,
    project_id: int,
    omero_client: OmeroClient,
):
    predictor = LungsPredictor(model)

    image = omero_client.download_image(image_id)

    try:
        roi, lungs_roi = predictor.compute_3d_roi(image)
    except:
        print(
            f"An error occured while computing the ROI in this image: ID={image_id}. Skipping...",
        )

    posted_image_id = omero_client.import_image_to_ds(
        roi, project_id, dataset_id, image_name
    )

    # Add tags
    roi_tag_id = omero_client.create_tag(project_id, "roi")
    omero_client.tag_image_with_tag(posted_image_id, tag_id=roi_tag_id)

    image_tags_list = find_image_tag(omero_client.get_image_tags(image_id))

    omero_client.copy_image_tags(
        src_image_id=image_id,
        dst_image_id=posted_image_id,
        exclude_tags=image_tags_list,
    )

    print("ROI detection workflow completed!")


def _compute_nnunet(
    model,
    image_name: str,
    image_id: int,
    dataset_id: int,
    project_id: int,
    omero_client: OmeroClient,
):
    predictor = TumorPredictor(model)

    image = omero_client.download_image(image_id)

    try:
        image_pred = predictor.predict(image)
    except:
        print(
            f"An error occured while computing the NNUNET prediction in this image: ID={image_id}."
        )
        return

    posted_image_id = omero_client.import_image_to_ds(
        image_pred, project_id, dataset_id, image_name
    )

    pred_tag_id = omero_client.create_tag(project_id, "raw_pred")
    omero_client.tag_image_with_tag(posted_image_id, tag_id=pred_tag_id)
    omero_client.copy_image_tags(
        src_image_id=image_id,
        dst_image_id=posted_image_id,
        exclude_tags=["roi"],
    )

    print("Segmentation workflow completed!")
