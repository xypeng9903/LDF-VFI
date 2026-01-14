import torch

#----------------------------------------------------------------------------
# Print summary table of module hierarchy.

@torch.no_grad()
def print_module_summary(module, max_nesting=3, skip_redundant=True):
    """Print a summary table of the module hierarchy showing only parameter counts.

    Args:
        module: The module to summarize
        max_nesting: Maximum depth of module nesting to include in the summary
        skip_redundant: Skip redundant entries with no parameters
    """
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    
    # Collect information about the module hierarchy
    entries = []
    
    for mod_name, mod in module.named_modules():
        display_name = mod_name if mod_name else '<top-level>'
        if mod is module or len(mod_name.split('.')) <= max_nesting:
            params = list(mod.parameters(recurse=True))
            entries.append({
                'name': display_name,
                'mod': mod,
                'params': params,
            })
    
    # Filter out redundant entries if requested
    if skip_redundant:
        entries = [e for e in entries if len(e['params'])]
    # Construct table
    rows = [['Module', 'Parameters']]
    rows += [['---', '---']]
    
    for e in entries:
        name = e['name'] # Use the specific name for this entry/path
        param_size = sum(t.numel() for t in e['params'])
        rows += [[
            name,
            str(param_size) if param_size else '-',
        ]]
    
    # Calculate the true total parameters of the module, correctly handling shared parameters
    param_total = sum(p.numel() for p in module.parameters())

    rows += [['---', '---']]
    if param_total > 1e9:
        param_total_str = f"{param_total:,} ({param_total / 1e9:.2f}B)"
    elif param_total > 1e6:
        param_total_str = f"{param_total:,} ({param_total / 1e6:.2f}M)"
    else:
        param_total_str = f"{param_total:,} ({param_total / 1e3:.2f}K)"
    rows += [['Total', param_total_str]]

    # Print table
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()
    return param_total

#----------------------------------------------------------------------------