import pandas as pd
from app import create_app


def client(tmp_path):
    path=tmp_path/'fires.csv'
    pd.DataFrame({"latitude":[1,2,3],"longitude":[4,5,6],"acq_date":["2024-01-01","2024-01-02","2024-01-03"],"frp":[1,2,3],"satellite":["A","A","B"],"confidence":["h","n","h"]}).to_csv(path,index=False)
    app=create_app(); app.config.update(TESTING=True,DATASET_PATH=str(path))
    return app.test_client()


def test_core_api_endpoints(tmp_path):
    app = client(tmp_path)
    assert app.get('/api/health').status_code == 200
    assert app.get('/api/schema').get_json()['records'] == 3
    assert app.post('/api/summary', json={'source': 'A'}).get_json()['total_detections'] == 2
    overview_res = app.post('/api/overview', json={})
    assert overview_res.status_code == 200
    overview_json = overview_res.get_json()
    assert "alerts" in overview_json
    assert isinstance(overview_json["alerts"], list)
    assert "kpis" in overview_json["summary"]
    assert len(overview_json["summary"]["kpis"]) > 0
    assert "detail" in overview_json["summary"]["kpis"][0]
    assert app.post('/api/map', json={}).get_json()['available'] is True
    geo_res = app.post('/api/geographic', json={})
    assert geo_res.status_code == 200
    geo_json = geo_res.get_json()
    assert geo_json['available'] is True
    assert len(geo_json['kpis']) > 0
    assert len(geo_json['clusters']) > 0
    assert 'lat_distribution' in geo_json
    assert 'quadrants' in geo_json
    sat_res = app.post('/api/satellites', json={})
    assert sat_res.status_code == 200
    sat_json = sat_res.get_json()
    assert sat_json['available'] is True
    assert len(sat_json['platforms']) > 0
    assert 'items' in sat_json
    frp_res = app.post('/api/frp', json={})
    assert frp_res.status_code == 200
    frp_json = frp_res.get_json()
    assert frp_json['available'] is True
    assert len(frp_json['kpis']) > 0
    assert len(frp_json['log_distribution']) > 0
    assert 'stats' in frp_json
    conf_res = app.post('/api/confidence', json={})
    assert conf_res.status_code == 200
    conf_json = conf_res.get_json()
    assert conf_json['available'] is True
    assert len(conf_json['kpis']) > 0
    assert len(conf_json['classes']) == 3
    assert 'actionable_rate' in conf_json
    assert 'items' in conf_json
    assert 'text/csv' in app.post('/api/download', json={}).content_type


def test_missing_dataset_is_friendly(tmp_path):
    app = create_app()
    app.config.update(TESTING=True, DATASET_PATH=str(tmp_path / 'missing.csv'))
    response = app.test_client().get('/api/health')
    assert response.status_code == 503


def test_alert_rules_crud(tmp_path):
    app = client(tmp_path)
    app.application.config["ALERTS_FILE"] = str(tmp_path / "alerts.json")

    # Create rule
    create_res = app.post('/api/alerts', json={"name": "High FRP Alert", "frp_min": 2.0})
    assert create_res.status_code == 201
    rule_id = create_res.get_json()['rule']['id']

    # List rules
    list_res = app.get('/api/alerts')
    assert len(list_res.get_json()['rules']) == 1

    # Check alert fires against dataset (FRP >= 2 has records 2 and 3)
    check_res = app.post('/api/alerts/check')
    assert check_res.status_code == 200
    firing = check_res.get_json()['firing']
    assert len(firing) == 1
    assert firing[0]['matched_count'] == 2

    # Toggle rule
    toggle_res = app.patch(f'/api/alerts/{rule_id}/toggle')
    assert toggle_res.status_code == 200
    assert toggle_res.get_json()['rule']['active'] is False

    # Delete rule
    del_res = app.delete(f'/api/alerts/{rule_id}')
    assert del_res.status_code == 200


def test_datasets_management(tmp_path):
    app = client(tmp_path)
    data_dir = tmp_path / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    app.application.config["DATA_DIRECTORY"] = str(data_dir)

    # Add a second dataset file
    file2 = data_dir / "second_dataset.csv"
    pd.DataFrame({"latitude": [10], "longitude": [20], "acq_date": ["2024-05-01"], "frp": [5]}).to_csv(file2, index=False)

    # 1. List datasets
    res = app.get('/api/datasets')
    assert res.status_code == 200
    datasets = res.get_json()['datasets']
    names = [d['name'] for d in datasets]
    assert "second_dataset.csv" in names

    # 2. Switch to second dataset
    sel_res = app.post('/api/datasets/select', json={"name": "second_dataset.csv"})
    assert sel_res.status_code == 200
    assert sel_res.get_json()['dataset']['records'] == 1

    # 3. Delete second dataset
    del_res = app.delete('/api/datasets/second_dataset.csv')
    assert del_res.status_code == 200
    assert not file2.exists()


