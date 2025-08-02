#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training example for MACAP with multi-agent system
"""

import asyncio
import logging
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from macap import (
    AgentConfig, ConfigManager,
    FewShotTrainer
)


async def main():
    """Main training function"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting MACAP training with multi-agent system")
    
    try:
        # Load configurations
        config_path = Path(__file__).parent.parent / "macap" / "config" / "agent_config.json"
        
        agent_config = ConfigManager.load_from_file(str(config_path))
        
        logger.info("✅ Configuration loaded")
        
        # Create trainer
        trainer = FewShotTrainer(agent_config)
        
        # Initialize trainer
        logger.info("⚙️ Initializing trainer...")
        await trainer.initialize()
        logger.info("✅ Trainer initialized")
        
        # Start training
        logger.info("🎯 Starting training...")
        await trainer.train()
        
        logger.info("🎉 Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())