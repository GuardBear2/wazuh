/*
 * Wazuh Vulnerability scanner - Scan Orchestrator
 * Copyright (C) 2015, Wazuh Inc.
 * January 22, 2025.
 *
 * This program is free software; you can redistribute it
 * and/or modify it under the terms of the GNU General Public
 * License (version 2) as published by the FSF - Free Software
 * Foundation.
 */

#ifndef _PACKAGE_ELEMENT_HPP
#define _PACKAGE_ELEMENT_HPP

#include "../../wcsModel/data.hpp"
#include "../../wcsModel/inventoryPackageHarvester.hpp"
#include "../../wcsModel/noData.hpp"

template<typename TContext>
class PackageElement final
{
public:
    // LCOV_EXCL_START
    /**
     * @brief Class destructor.
     *
     */
    ~PackageElement() = default;
    // LCOV_EXCL_STOP

    static DataHarvester<InventoryPackageHarvester> build(TContext* data)
    {
        DataHarvester<InventoryPackageHarvester> element;
        element.id = data->agentId();
        element.id += "_";
        element.id += data->packageItemId();
        element.operation = "INSERTED";
        element.data.agent.id = data->agentId();
        element.data.agent.name = data->agentName();
        element.data.agent.version = data->agentVersion();
        element.data.agent.ip = data->agentIp();

        element.data.package.architecture = data->packageArchitecture();
        element.data.package.name = data->packageName();
        element.data.package.version = data->packageVersion();
        element.data.package.vendor = data->packageVendor();
        element.data.package.installed = data->packageInstallTime().compare(" ") == 0 ? "" : data->packageInstallTime();
        element.data.package.size = data->packageSize();
        element.data.package.type = data->packageFormat();
        element.data.package.description = data->packageDescription();
        element.data.package.path = data->packageLocation();

        return element;
    }

    static NoDataHarvester deleteElement(TContext* data)
    {
        NoDataHarvester element;
        element.operation = "DELETED";
        element.id = data->agentId();
        element.id += "_";
        element.id += data->packageItemId();
        return element;
    }
};

#endif // _PACKAGE_ELEMENT_HPP
