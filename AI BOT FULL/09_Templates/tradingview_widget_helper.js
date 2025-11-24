// TradingView Widget Helper Functions
// Helper functions to add Order Blocks, FVGs, and Trades to TradingView Widget

function addOrderBlocksToTradingView(widget, orderBlocks) {
    if (!widget || !orderBlocks || orderBlocks.length === 0) return;
    
    console.log(`[TV] Adding ${orderBlocks.length} Order Blocks to TradingView Widget`);
    
    orderBlocks.forEach((ob, index) => {
        try {
            const time = ob.time || Math.floor(Date.now() / 1000);
            const high = ob.high || 0;
            const low = ob.low || 0;
            const type = ob.type || 'Bullish';
            
            // Create rectangle shape for Order Block
            widget.chart().createShape(
                [
                    { time: time, price: high },
                    { time: time, price: low }
                ],
                {
                    shape: 'rectangle',
                    lock: true,
                    disableSelection: true,
                    disableSave: true,
                    text: type === 'Bullish' ? 'OB↑' : 'OB↓',
                    overrides: {
                        backgroundColor: type === 'Bullish' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                        borderColor: type === 'Bullish' ? '#10b981' : '#ef4444',
                        borderWidth: 2,
                        lineStyle: 0, // Solid
                    }
                }
            );
        } catch (e) {
            console.warn(`[TV] Error adding Order Block ${index}:`, e);
        }
    });
}

function addFVGsToTradingView(widget, fvgs) {
    if (!widget || !fvgs || fvgs.length === 0) return;
    
    console.log(`[TV] Adding ${fvgs.length} FVGs to TradingView Widget`);
    
    fvgs.forEach((fvg, index) => {
        try {
            const time = fvg.time || Math.floor(Date.now() / 1000);
            const top = fvg.top || 0;
            const bottom = fvg.bottom || 0;
            const type = fvg.type || 'Bullish';
            
            // Create rectangle shape for FVG (dashed)
            widget.chart().createShape(
                [
                    { time: time, price: top },
                    { time: time, price: bottom }
                ],
                {
                    shape: 'rectangle',
                    lock: true,
                    disableSelection: true,
                    disableSave: true,
                    text: type === 'Bullish' ? 'FVG↑' : 'FVG↓',
                    overrides: {
                        backgroundColor: type === 'Bullish' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                        borderColor: type === 'Bullish' ? '#10b981' : '#ef4444',
                        borderWidth: 1,
                        lineStyle: 2, // Dashed
                    }
                }
            );
        } catch (e) {
            console.warn(`[TV] Error adding FVG ${index}:`, e);
        }
    });
}

function addTradesToTradingView(widget, trades) {
    if (!widget || !trades || trades.length === 0) return;
    
    console.log(`[TV] Adding ${trades.length} Trades to TradingView Widget`);
    
    trades.forEach((trade, index) => {
        try {
            const time = trade.entry_time || trade.time || Math.floor(Date.now() / 1000);
            const price = trade.entry_price || trade.price || 0;
            const type = trade.type || trade.direction || 'BUY';
            
            // Create marker for trade entry
            widget.chart().createShape(
                [{ time: time, price: price }],
                {
                    shape: 'circle',
                    lock: true,
                    disableSelection: true,
                    disableSave: true,
                    text: type === 'BUY' ? '↑' : '↓',
                    overrides: {
                        backgroundColor: type === 'BUY' ? '#10b981' : '#ef4444',
                        borderColor: type === 'BUY' ? '#10b981' : '#ef4444',
                        borderWidth: 2,
                    }
                }
            );
        } catch (e) {
            console.warn(`[TV] Error adding Trade ${index}:`, e);
        }
    });
}


