import pandas as pd
from app import create_app
from app.services.dataset_service import get_dataset


def test_schema_detection_and_filtering(tmp_path):
    path = tmp_path / "fires.csv"
    pd.DataFrame({"latitude":[12,13],"longitude":[78,79],"acq_date":["2024-01-01","2024-02-01"],"frp":[3.5,10],"satellite":["A","B"],"confidence":["h","n"]}).to_csv(path,index=False)
    app=create_app(); app.config.update(TESTING=True, DATASET_PATH=str(path))
    with app.app_context():
        service=get_dataset()
        assert service.columns["latitude"] == "latitude"
        assert len(service.filter({"source":"A"})) == 1
        assert service.schema()["records"] == 2
