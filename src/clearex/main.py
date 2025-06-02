import typer

app = typer.Typer(help="Command line interface for ClearEx")

CLEAR_EX_LOGO = r"""
 ░▒▓██████▓▒░░▒▓█▓▒░      ░▒▓████████▓▒░░▒▓██████▓▒░░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓██████▓▒░ ░▒▓████████▓▒░▒▓███████▓▒░░▒▓██████▓▒░  ░▒▓██████▓▒░  
░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
 ░▒▓██████▓▒░░▒▓████████▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░ 
"""

@app.command()
def main():
    """Run the ClearEx command line interface."""
    typer.echo(CLEAR_EX_LOGO)
    typer.echo("Welcome to ClearEx!")

    operation = typer.prompt(
        "What type of image operation would you like to perform?",
        default="registration",
    )

    if operation.lower() != "registration":
        typer.echo("Only the registration routine is currently available.")
        raise typer.Exit(code=1)

    reference = typer.prompt("Reference image location")
    moving = typer.prompt("Image to register")
    output = typer.prompt("Location to save the data")

    typer.echo("Starting registration...")
    # Placeholder for future registration call
    typer.echo(f"Reference: {reference}")
    typer.echo(f"Moving: {moving}")
    typer.echo(f"Output directory: {output}")

    typer.echo("Done.")


if __name__ == "__main__":
    app()
