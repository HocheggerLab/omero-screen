#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.

# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.
# ------------------------------------------------------------------------------

"""Script to send an email."""

import argparse
import getpass


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Program to send a test e-mail to the given addresses."
    )
    parser.add_argument("address", nargs="+", help="e-mail recipient address")

    group = parser.add_argument_group("Message")
    group.add_argument(
        "--sender",
        dest="sender",
        default=getpass.getuser() + "@sussex.ac.uk",
        help="e-mail sender [%(default)s]",
    )
    group.add_argument(
        "-s",
        "--subject",
        dest="subject",
        default="SMTP e-mail test",
        help="e-mail subject [%(default)s]",
    )
    group.add_argument(
        "-m",
        "--message",
        dest="message",
        default=("This is a test e-mail message."),
        help="e-mail message [%(default)s]",
    )
    group.add_argument(
        "-f",
        "--filename",
        dest="file",
        metavar="FILE",
        default=None,
        help="e-mail message text file (overrides other arguments)",
    )
    group.add_argument(
        "--smtp",
        action="store_true",
        default=False,
        help="Use SMTP lib (default is sendmail)",
    )

    args = parser.parse_args()

    if args.file:
        with open(args.file) as file:
            message = file.read()
    else:
        message = f"""Subject: {args.subject}

{args.message}
"""

    if args.smtp:
        from smtplib import SMTP

        with SMTP("localhost") as smtp:
            # This method will return normally if the mail is accepted for at
            # least one recipient (i.e. someone should get your mail).
            # Otherwise it will raise an exception.
            smtp.sendmail(args.sender, args.address, message)
    else:
        # Find full path to sendmail
        from shutil import which

        sm = which("sendmail")
        if sm is None:
            raise Exception("No sendmail on the path")

        # Execute sendmail
        from subprocess import PIPE, STDOUT, Popen, TimeoutExpired

        p = Popen(
            [sm, "-t"], stdin=PIPE, stdout=PIPE, stderr=STDOUT, text=True
        )
        try:
            message = f"""From: {args.sender}
To: {",".join(args.address)}
{message}
"""
            outs, _ = p.communicate(message, timeout=15)
        except TimeoutExpired:
            p.kill()
            outs, _ = p.communicate()

        if p.returncode:
            print(f"ERROR: exit code={p.returncode}; {outs}")


# Standard boilerplate to call the function to begin the program.
if __name__ == "__main__":
    _main()
