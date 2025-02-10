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

#ifndef _REGISTRY_VALUE_ELEMENT_HPP
#define _REGISTRY_VALUE_ELEMENT_HPP

#include "../../wcsModel/data.hpp"
#include "../../wcsModel/fimRegistryHarvester.hpp"
#include "../../wcsModel/noData.hpp"
#include "stringHelper.h"

template<typename TContext>
class RegistryValueElement final
{
public:
    // LCOV_EXCL_START
    /**
     * @brief Class destructor.
     *
     */
    ~RegistryValueElement() = default;
    // LCOV_EXCL_STOP

    static DataHarvester<FimRegistryInventoryHarvester> build(TContext* data)
    {
        DataHarvester<FimRegistryInventoryHarvester> element;
        element.id = data->agentId();
        element.id += "_";
        element.id += data->path();
        element.operation = "INSERTED";

        element.data.agent.id = data->agentId();
        element.data.agent.name = data->agentName();
        element.data.agent.version = data->agentVersion();
        element.data.agent.ip = data->agentIp();

        element.data.registry.hive = data->hive();
        element.data.registry.key = data->key();
        element.data.registry.path = data->path();
        element.data.registry.value = data->valueName();

        element.data.registry.data.hash.md5 = data->md5();
        element.data.registry.data.hash.sha1 = data->sha1();
        element.data.registry.data.hash.sha256 = data->sha256();
        element.data.registry.data.type = data->valueType();

        return element;
    }

    static NoDataHarvester deleteElement(TContext* data)
    {
        NoDataHarvester element;
        element.operation = "DELETED";
        element.id = data->agentId();
        element.id += "_";
        element.id += data->path();

        return element;
    }
};

#endif // _REGISTRY_VALUE_ELEMENT_HPP
