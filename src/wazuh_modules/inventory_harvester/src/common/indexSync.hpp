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

#ifndef _INDEX_SYNC_HPP
#define _INDEX_SYNC_HPP

#include "chainOfResponsability.hpp"
#include "indexerConnector.hpp"
#include <map>
#include <memory>

template<typename TContext>
class IndexSync final : public AbstractHandler<std::shared_ptr<TContext>>
{
    const std::map<typename TContext::AffectedComponentType, std::unique_ptr<IndexerConnector>, std::less<>>&
        m_indexerConnectorInstances;

public:
    // LCOV_EXCL_START
    /**
     * @brief Class destructor.
     *
     */
    ~IndexSync() = default;

    explicit IndexSync(
        const std::map<typename TContext::AffectedComponentType, std::unique_ptr<IndexerConnector>, std::less<>>&
            indexerConnectorInstances)
        : m_indexerConnectorInstances(indexerConnectorInstances)
    {
    }
    // LCOV_EXCL_STOP

    /**
     * @brief Handles request and passes control to the next step of the chain.
     *
     * @param data Scan context.
     * @return std::shared_ptr<ScanContext> Abstract handler.
     */
    std::shared_ptr<TContext> handleRequest(std::shared_ptr<TContext> data) override
    {
        m_indexerConnectorInstances.at(data->affectedComponentType())->sync(std::string(data->agentId()));
        return AbstractHandler<std::shared_ptr<TContext>>::handleRequest(std::move(data));
    }
};

#endif // _INDEX_SYNC_HPP
