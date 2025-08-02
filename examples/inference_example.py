#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference example for MACAP with multi-agent system
"""

import asyncio
import logging
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from macap import (
    AgentConfig, create_agent_graph,
    create_cap_model, create_data_loader
)


async def main():
    """Main inference function"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🔍 Starting MACAP inference example")
    
    try:
        # Create configuration
        agent_config = AgentConfig()
        
        # Create agent graph
        logger.info("🤖 Creating agent graph...")
        agent_graph = await create_agent_graph(agent_config)
        logger.info("✅ Agent graph created")
        
        # Create model
        logger.info("🧠 Creating model...")
        model = create_cap_model(
            n_way=5,
            enable_agent_enhancement=True
        )
        logger.info("✅ Model created")
        
        # Prepare test data
        test_instances = [
            {
                'token': ['Elon', 'Musk', 'is', 'the', 'CEO', 'of', 'Tesla'],
                'h': {'name': 'Elon Musk', 'pos': [0, 1]},
                't': {'name': 'Tesla', 'pos': [6, 6]},
                'relation': 'CEO_of'
            },
            {
                'token': ['Albert', 'Einstein', 'was', 'born', 'in', 'Germany'],
                'h': {'name': 'Albert Einstein', 'pos': [0, 1]},
                't': {'name': 'Germany', 'pos': [5, 5]},
                'relation': 'born_in'
            }
        ]
        
        # Agent processing
        logger.info("🧠 Processing with agents...")
        agent_result = await agent_graph.execute(test_instances, n_way=2, k_shot=1)
        
        if agent_result.get('error_info'):
            logger.warning(f"⚠️ Agent processing had errors: {agent_result['error_info']['message']}")
        else:
            logger.info("✅ Agent processing completed")
            logger.info(f"  - Aligned concepts: {len(agent_result.get('aligned_concepts', []))}")
            logger.info(f"  - Meta relations: {len(agent_result.get('meta_relations', []))}")
        
        # Cleanup
        await agent_graph.cleanup()
        logger.info("✅ Inference example completed")
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())