name: Build Docker Images for Package Building

on:
  workflow_dispatch:
    inputs:
      tag:
        description: 'Tag for the Docker images'
        required: true
        default: '5.0.0'

jobs:
  build-deb-builder:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
          
      - name: Prepare build context
        run: |
          mkdir -p /tmp/deb-build-context
          cp packages/debs/amd64/Dockerfile /tmp/deb-build-context/
          cp src/engine/vcpkg.json /tmp/deb-build-context/
          cp src/engine/vcpkg-configuration.json /tmp/deb-build-context/
          cp packages/build.sh /tmp/deb-build-context/build.sh
          cp packages/debs/utils/helper_function.sh /tmp/deb-build-context/
          cp packages/debs/utils/gen_permissions.sh /tmp/deb-build-context/

      - name: Build and push DEB manager builder
        uses: docker/build-push-action@v5
        with:
          context: /tmp/deb-build-context
          push: true
          tags: ghcr.io/guardbear2/pkg_deb_manager_builder_amd64:${{ github.event.inputs.tag }}

  build-rpm-builder:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
          
      - name: Prepare build context
        run: |
          mkdir -p /tmp/rpm-build-context
          cp packages/rpms/amd64/Dockerfile /tmp/rpm-build-context/
          cp src/engine/vcpkg.json /tmp/rpm-build-context/
          cp src/engine/vcpkg-configuration.json /tmp/rpm-build-context/
          cp packages/build.sh /tmp/rpm-build-context/build.sh
          cp packages/rpms/utils/helper_function.sh /tmp/rpm-build-context/

      - name: Build and push RPM manager builder
        uses: docker/build-push-action@v5
        with:
          context: /tmp/rpm-build-context
          push: true
          tags: ghcr.io/guardbear2/pkg_rpm_manager_builder_amd64:${{ github.event.inputs.tag }} 