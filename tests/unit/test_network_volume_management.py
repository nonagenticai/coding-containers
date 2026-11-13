"""Tests for services/docker_services/network_volume_management.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import docker.errors

from src.services.docker_services.network_volume_management import NetworkVolumeManager

# ===========================================================================
# Volume Management Tests
# ===========================================================================


class TestEnsureVolumeExists:
    def test_no_client(self):
        mgr = NetworkVolumeManager(None, "testproject")
        assert mgr.ensure_volume_exists("vol1") is False

    def test_volume_already_exists(self):
        client = MagicMock()
        client.volumes.get.return_value = MagicMock()
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.ensure_volume_exists("vol1") is True
        client.volumes.get.assert_called_once_with("vol1")
        client.volumes.create.assert_not_called()

    def test_volume_created(self):
        client = MagicMock()
        client.volumes.get.side_effect = docker.errors.NotFound("not found")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.ensure_volume_exists("vol1") is True
        client.volumes.create.assert_called_once()
        call_kwargs = client.volumes.create.call_args[1]
        assert call_kwargs["name"] == "vol1"
        assert call_kwargs["driver"] == "local"
        assert call_kwargs["labels"]["com.docker.compose.project"] == "testproject"
        assert call_kwargs["labels"]["created_by"] == "code_graph_helper"

    def test_volume_created_with_custom_driver_and_labels(self):
        client = MagicMock()
        client.volumes.get.side_effect = docker.errors.NotFound("not found")
        mgr = NetworkVolumeManager(client, "testproject")

        result = mgr.ensure_volume_exists("vol1", driver="nfs", labels={"env": "prod"})
        assert result is True
        call_kwargs = client.volumes.create.call_args[1]
        assert call_kwargs["driver"] == "nfs"
        assert call_kwargs["labels"]["env"] == "prod"

    def test_volume_creation_fails(self):
        client = MagicMock()
        client.volumes.get.side_effect = docker.errors.NotFound("not found")
        client.volumes.create.side_effect = docker.errors.DockerException("create fail")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.ensure_volume_exists("vol1") is False

    def test_generic_get_error(self):
        client = MagicMock()
        client.volumes.get.side_effect = docker.errors.DockerException("unexpected")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.ensure_volume_exists("vol1") is False


class TestRemoveVolume:
    def test_no_client(self):
        mgr = NetworkVolumeManager(None, "testproject")
        assert mgr.remove_volume("vol1") is False

    def test_success(self):
        client = MagicMock()
        mock_vol = MagicMock()
        client.volumes.get.return_value = mock_vol
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.remove_volume("vol1", force=True) is True
        mock_vol.remove.assert_called_once_with(force=True)

    def test_not_found_returns_true(self):
        client = MagicMock()
        client.volumes.get.side_effect = docker.errors.NotFound("not found")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.remove_volume("vol1") is True

    def test_api_error_volume_in_use(self):
        client = MagicMock()
        mock_vol = MagicMock()
        client.volumes.get.return_value = mock_vol
        mock_vol.remove.side_effect = docker.errors.APIError("volume is in use")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.remove_volume("vol1", force=False) is False

    def test_api_error_other(self):
        client = MagicMock()
        mock_vol = MagicMock()
        client.volumes.get.return_value = mock_vol
        mock_vol.remove.side_effect = docker.errors.APIError("some other error")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.remove_volume("vol1", force=True) is False

    def test_generic_exception(self):
        client = MagicMock()
        client.volumes.get.side_effect = docker.errors.DockerException("unexpected")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.remove_volume("vol1") is False


class TestListVolumes:
    def test_no_client(self):
        mgr = NetworkVolumeManager(None, "testproject")
        assert mgr.list_volumes() == []

    def test_success(self):
        client = MagicMock()
        mock_vol = MagicMock()
        mock_vol.name = "test-vol"
        mock_vol.attrs = {
            "Driver": "local",
            "Mountpoint": "/var/lib/docker/volumes/test-vol/_data",
            "CreatedAt": "2024-01-01T00:00:00Z",
            "Labels": {"env": "dev"},
            "Scope": "local",
        }
        client.volumes.list.return_value = [mock_vol]
        mgr = NetworkVolumeManager(client, "testproject")

        result = mgr.list_volumes(filters={"label": "env=dev"})
        assert len(result) == 1
        assert result[0]["name"] == "test-vol"
        assert result[0]["driver"] == "local"
        assert result[0]["labels"] == {"env": "dev"}

    def test_no_labels(self):
        client = MagicMock()
        mock_vol = MagicMock()
        mock_vol.name = "vol"
        mock_vol.attrs = {"Labels": None}
        client.volumes.list.return_value = [mock_vol]
        mgr = NetworkVolumeManager(client, "testproject")

        result = mgr.list_volumes()
        assert result[0]["labels"] == {}

    def test_exception(self):
        client = MagicMock()
        client.volumes.list.side_effect = docker.errors.DockerException("list fail")
        mgr = NetworkVolumeManager(client, "testproject")
        assert mgr.list_volumes() == []


class TestGetVolumeInfo:
    def test_no_client(self):
        mgr = NetworkVolumeManager(None, "testproject")
        assert mgr.get_volume_info("vol1") is None

    def test_success(self):
        client = MagicMock()
        mock_vol = MagicMock()
        mock_vol.name = "vol1"
        mock_vol.attrs = {
            "Driver": "local",
            "Mountpoint": "/data",
            "CreatedAt": "2024-01-01",
            "Labels": {"env": "dev"},
            "Scope": "local",
            "Options": {"type": "nfs"},
        }
        client.volumes.get.return_value = mock_vol
        mgr = NetworkVolumeManager(client, "testproject")

        result = mgr.get_volume_info("vol1")
        assert result is not None
        assert result["name"] == "vol1"
        assert result["driver"] == "local"
        assert result["options"] == {"type": "nfs"}

    def test_not_found(self):
        client = MagicMock()
        client.volumes.get.side_effect = docker.errors.NotFound("not found")
        mgr = NetworkVolumeManager(client, "testproject")
        assert mgr.get_volume_info("missing") is None

    def test_exception(self):
        client = MagicMock()
        client.volumes.get.side_effect = docker.errors.DockerException("unexpected")
        mgr = NetworkVolumeManager(client, "testproject")
        assert mgr.get_volume_info("broken") is None


class TestPruneVolumes:
    def test_no_client(self):
        mgr = NetworkVolumeManager(None, "testproject")
        result = mgr.prune_volumes()
        assert result == {"volumes_deleted": [], "space_reclaimed": 0}

    def test_success(self):
        client = MagicMock()
        client.volumes.prune.return_value = {
            "VolumesDeleted": ["v1", "v2"],
            "SpaceReclaimed": 5000,
        }
        mgr = NetworkVolumeManager(client, "testproject")

        result = mgr.prune_volumes(filters={"label": "cleanup=true"})
        assert result["volumes_deleted"] == ["v1", "v2"]
        assert result["space_reclaimed"] == 5000

    def test_none_deleted(self):
        client = MagicMock()
        client.volumes.prune.return_value = {
            "VolumesDeleted": None,
            "SpaceReclaimed": 0,
        }
        mgr = NetworkVolumeManager(client, "testproject")

        result = mgr.prune_volumes()
        assert result["volumes_deleted"] == []

    def test_exception(self):
        client = MagicMock()
        client.volumes.prune.side_effect = docker.errors.DockerException("prune fail")
        mgr = NetworkVolumeManager(client, "testproject")

        result = mgr.prune_volumes()
        assert result == {"volumes_deleted": [], "space_reclaimed": 0}


# ===========================================================================
# Network Management Tests
# ===========================================================================


class TestEnsureNetworkExists:
    def test_no_client(self):
        mgr = NetworkVolumeManager(None, "testproject")
        assert mgr.ensure_network_exists("net1") is False

    def test_network_already_exists(self):
        client = MagicMock()
        client.networks.get.return_value = MagicMock()
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.ensure_network_exists("net1") is True
        client.networks.create.assert_not_called()

    def test_network_created(self):
        client = MagicMock()
        client.networks.get.side_effect = docker.errors.NotFound("not found")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.ensure_network_exists("net1") is True
        client.networks.create.assert_called_once()
        call_kwargs = client.networks.create.call_args[1]
        assert call_kwargs["name"] == "net1"
        assert call_kwargs["driver"] == "bridge"
        assert call_kwargs["labels"]["com.docker.compose.project"] == "testproject"

    def test_network_created_custom_driver(self):
        client = MagicMock()
        client.networks.get.side_effect = docker.errors.NotFound("not found")
        mgr = NetworkVolumeManager(client, "testproject")

        result = mgr.ensure_network_exists(
            "net1",
            driver="overlay",
            labels={"env": "prod"},
            options={"encrypted": "true"},
        )
        assert result is True
        call_kwargs = client.networks.create.call_args[1]
        assert call_kwargs["driver"] == "overlay"
        assert call_kwargs["options"] == {"encrypted": "true"}
        assert call_kwargs["labels"]["env"] == "prod"

    def test_network_creation_fails(self):
        client = MagicMock()
        client.networks.get.side_effect = docker.errors.NotFound("not found")
        client.networks.create.side_effect = docker.errors.DockerException("create fail")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.ensure_network_exists("net1") is False

    def test_generic_get_error(self):
        client = MagicMock()
        client.networks.get.side_effect = docker.errors.DockerException("unexpected")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.ensure_network_exists("net1") is False


class TestRemoveNetwork:
    def test_no_client(self):
        mgr = NetworkVolumeManager(None, "testproject")
        assert mgr.remove_network("net1") is False

    def test_success(self):
        client = MagicMock()
        mock_net = MagicMock()
        client.networks.get.return_value = mock_net
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.remove_network("net1") is True
        mock_net.remove.assert_called_once()

    def test_not_found_returns_true(self):
        client = MagicMock()
        client.networks.get.side_effect = docker.errors.NotFound("not found")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.remove_network("net1") is True

    def test_api_error_active_endpoints(self):
        client = MagicMock()
        mock_net = MagicMock()
        client.networks.get.return_value = mock_net
        mock_net.remove.side_effect = docker.errors.APIError("has active endpoints")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.remove_network("net1") is False

    def test_api_error_other(self):
        client = MagicMock()
        mock_net = MagicMock()
        client.networks.get.return_value = mock_net
        mock_net.remove.side_effect = docker.errors.APIError("some other error")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.remove_network("net1") is False

    def test_generic_exception(self):
        client = MagicMock()
        client.networks.get.side_effect = docker.errors.DockerException("unexpected")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.remove_network("net1") is False


class TestListNetworks:
    def test_no_client(self):
        mgr = NetworkVolumeManager(None, "testproject")
        assert mgr.list_networks() == []

    def test_success(self):
        client = MagicMock()
        mock_net = MagicMock()
        mock_net.id = "net123"
        mock_net.name = "testnet"
        mock_net.attrs = {
            "Driver": "bridge",
            "Scope": "local",
            "Created": "2024-01-01",
            "Labels": {"env": "dev"},
            "Options": {},
            "Containers": {"c1": {}, "c2": {}},
        }
        client.networks.list.return_value = [mock_net]
        mgr = NetworkVolumeManager(client, "testproject")

        result = mgr.list_networks(filters={"driver": "bridge"})
        assert len(result) == 1
        assert result[0]["id"] == "net123"
        assert result[0]["name"] == "testnet"
        assert len(result[0]["containers"]) == 2

    def test_empty_attrs(self):
        client = MagicMock()
        mock_net = MagicMock()
        mock_net.id = "n1"
        mock_net.name = "empty"
        mock_net.attrs = {"Labels": None, "Options": None, "Containers": {}}
        client.networks.list.return_value = [mock_net]
        mgr = NetworkVolumeManager(client, "testproject")

        result = mgr.list_networks()
        assert result[0]["labels"] == {}
        assert result[0]["options"] == {}
        assert result[0]["containers"] == []

    def test_exception(self):
        client = MagicMock()
        client.networks.list.side_effect = docker.errors.DockerException("list fail")
        mgr = NetworkVolumeManager(client, "testproject")
        assert mgr.list_networks() == []


class TestGetNetworkInfo:
    def test_no_client(self):
        mgr = NetworkVolumeManager(None, "testproject")
        assert mgr.get_network_info("net1") is None

    def test_success(self):
        client = MagicMock()
        mock_net = MagicMock()
        mock_net.id = "net123"
        mock_net.name = "testnet"
        mock_net.attrs = {
            "Driver": "bridge",
            "Scope": "local",
            "Created": "2024-01-01",
            "Labels": {"env": "dev"},
            "Options": {},
            "IPAM": {"Config": [{"Subnet": "172.17.0.0/16"}]},
            "Containers": {"c1": {"Name": "test"}},
            "Internal": False,
            "Attachable": True,
        }
        client.networks.get.return_value = mock_net
        mgr = NetworkVolumeManager(client, "testproject")

        result = mgr.get_network_info("testnet")
        assert result is not None
        assert result["id"] == "net123"
        assert result["attachable"] is True
        assert result["internal"] is False

    def test_not_found(self):
        client = MagicMock()
        client.networks.get.side_effect = docker.errors.NotFound("not found")
        mgr = NetworkVolumeManager(client, "testproject")
        assert mgr.get_network_info("missing") is None

    def test_exception(self):
        client = MagicMock()
        client.networks.get.side_effect = docker.errors.DockerException("unexpected")
        mgr = NetworkVolumeManager(client, "testproject")
        assert mgr.get_network_info("broken") is None


class TestConnectContainerToNetwork:
    def test_no_client(self):
        mgr = NetworkVolumeManager(None, "testproject")
        assert mgr.connect_container_to_network("c1", "net1") is False

    def test_success(self):
        client = MagicMock()
        mock_net = MagicMock()
        client.networks.get.return_value = mock_net
        mgr = NetworkVolumeManager(client, "testproject")

        result = mgr.connect_container_to_network("container1", "net1", aliases=["alias1"])
        assert result is True
        mock_net.connect.assert_called_once_with("container1", aliases=["alias1"])

    def test_not_found(self):
        client = MagicMock()
        client.networks.get.side_effect = docker.errors.NotFound("not found")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.connect_container_to_network("c1", "missing") is False

    def test_already_connected(self):
        client = MagicMock()
        mock_net = MagicMock()
        client.networks.get.return_value = mock_net
        mock_net.connect.side_effect = docker.errors.APIError("already exists in network")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.connect_container_to_network("c1", "net1") is True

    def test_api_error_other(self):
        client = MagicMock()
        mock_net = MagicMock()
        client.networks.get.return_value = mock_net
        mock_net.connect.side_effect = docker.errors.APIError("something else")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.connect_container_to_network("c1", "net1") is False

    def test_generic_exception(self):
        client = MagicMock()
        client.networks.get.side_effect = docker.errors.DockerException("unexpected")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.connect_container_to_network("c1", "net1") is False


class TestDisconnectContainerFromNetwork:
    def test_no_client(self):
        mgr = NetworkVolumeManager(None, "testproject")
        assert mgr.disconnect_container_from_network("c1", "net1") is False

    def test_success(self):
        client = MagicMock()
        mock_net = MagicMock()
        client.networks.get.return_value = mock_net
        mgr = NetworkVolumeManager(client, "testproject")

        result = mgr.disconnect_container_from_network("c1", "net1", force=True)
        assert result is True
        mock_net.disconnect.assert_called_once_with("c1", force=True)

    def test_not_found_returns_true(self):
        client = MagicMock()
        client.networks.get.side_effect = docker.errors.NotFound("not found")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.disconnect_container_from_network("c1", "net1") is True

    def test_api_error(self):
        client = MagicMock()
        mock_net = MagicMock()
        client.networks.get.return_value = mock_net
        mock_net.disconnect.side_effect = docker.errors.APIError("api error")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.disconnect_container_from_network("c1", "net1") is False

    def test_generic_exception(self):
        client = MagicMock()
        client.networks.get.side_effect = docker.errors.DockerException("unexpected")
        mgr = NetworkVolumeManager(client, "testproject")

        assert mgr.disconnect_container_from_network("c1", "net1") is False


class TestPruneNetworks:
    def test_no_client(self):
        mgr = NetworkVolumeManager(None, "testproject")
        result = mgr.prune_networks()
        assert result == {"networks_deleted": []}

    def test_success(self):
        client = MagicMock()
        client.networks.prune.return_value = {
            "NetworksDeleted": ["net1", "net2"],
        }
        mgr = NetworkVolumeManager(client, "testproject")

        result = mgr.prune_networks(filters={"label": "cleanup=true"})
        assert result["networks_deleted"] == ["net1", "net2"]

    def test_none_deleted(self):
        client = MagicMock()
        client.networks.prune.return_value = {"NetworksDeleted": None}
        mgr = NetworkVolumeManager(client, "testproject")

        result = mgr.prune_networks()
        assert result["networks_deleted"] == []

    def test_exception(self):
        client = MagicMock()
        client.networks.prune.side_effect = docker.errors.DockerException("prune fail")
        mgr = NetworkVolumeManager(client, "testproject")

        result = mgr.prune_networks()
        assert result == {"networks_deleted": []}


# ===========================================================================
# Utility Method Tests
# ===========================================================================


class TestCreateContainerMounts:
    @patch("src.services.docker_services.network_volume_management.docker")
    def test_creates_mounts(self, mock_docker_mod):
        client = MagicMock()
        mgr = NetworkVolumeManager(client, "myproject")

        with patch.object(mgr, "ensure_volume_exists", return_value=True) as mock_ensure:
            mgr.create_container_mounts("test-container")

        assert mock_ensure.call_count == 2
        mock_ensure.assert_any_call("test-container-data")
        mock_ensure.assert_any_call("myproject_shared_code_graphs")

        assert mock_docker_mod.types.Mount.call_count == 2


class TestSetupProjectInfrastructure:
    def test_no_client(self):
        mgr = NetworkVolumeManager(None, "testproject")
        assert mgr.setup_project_infrastructure() is False

    def test_success(self):
        client = MagicMock()
        mgr = NetworkVolumeManager(client, "testproject")

        with (
            patch.object(mgr, "ensure_network_exists", return_value=True) as mock_net,
            patch.object(mgr, "ensure_volume_exists", return_value=True) as mock_vol,
        ):
            result = mgr.setup_project_infrastructure()

        assert result is True
        mock_net.assert_called_once_with("workflow-system-network")
        mock_vol.assert_called_once_with("testproject_shared_code_graphs")

    def test_network_fails(self):
        client = MagicMock()
        mgr = NetworkVolumeManager(client, "testproject")

        with (
            patch.object(mgr, "ensure_network_exists", return_value=False),
            patch.object(mgr, "ensure_volume_exists", return_value=True),
        ):
            result = mgr.setup_project_infrastructure()

        assert result is False

    def test_volume_fails(self):
        client = MagicMock()
        mgr = NetworkVolumeManager(client, "testproject")

        with (
            patch.object(mgr, "ensure_network_exists", return_value=True),
            patch.object(mgr, "ensure_volume_exists", return_value=False),
        ):
            result = mgr.setup_project_infrastructure()

        assert result is False

    def test_exception(self):
        client = MagicMock()
        mgr = NetworkVolumeManager(client, "testproject")

        with patch.object(
            mgr, "ensure_network_exists", side_effect=docker.errors.DockerException("boom")
        ):
            result = mgr.setup_project_infrastructure()

        assert result is False
