from google.cloud import secretmanager


def create_secret(project_id, secret_id):
    """
    usage:
    >>> project_id = 'at-ml-platform'
    >>> secret_id = 'coinbase-api'
    >>> create_secret(project_id, secret_id)

    >>> create_secret(project_id, 'coinbase-key')
    >>> create_secret(project_id, 'coinbase-secret')
    >>> create_secret(project_id, 'coinbase-pass')

    """
    client = secretmanager.SecretManagerServiceClient()
    parent = f"projects/{project_id}"

    # Create the secret.
    response = client.create_secret(
        request={
            "parent": parent,
            "secret_id": secret_id,
            "secret": {"replication": {"automatic": {}}},
        }
    )
    return response


def add_secret_version(project_id, secret_id, payload):
    """
    Add a new secret version to the given secret with the provided payload.
    usage:
    >>> project_id = 'at-ml-platform'
    >>> secret_id = 'coinbase-api'

    """
    client = secretmanager.SecretManagerServiceClient()
    parent = client.secret_path(project_id, secret_id)
    payload = payload.encode("UTF-8")
    response = client.add_secret_version(
        request={"parent": parent, "payload": {"data": payload}}
    )
    return response


def access_secret_version(project_id, secret_id, version_id):
    """
    Access the payload for the given secret version if one exists. The version
    can be a version number as a string (e.g. "5") or an alias (e.g. "latest").
    usage:
    >>> project_id = 'at-ml-platform'
    >>> secret_id = 'coinbase-api'
    >>> version_id = 'latest'
    """

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    payload = response.payload.data.decode("UTF-8")
    return payload
