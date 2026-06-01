/**
 * Tests for the /latent page.
 *
 * Coverage:
 *   - Page renders three panes (raw, reconstruction, error-map)
 *   - Null reconstruction field shows a placeholder, not an error
 *   - Navigation link to main page exists on latent page
 *   - Navigation link to latent page exists on main page
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { useVisualizerSocket } from "@/hooks/useVisualizerSocket";

// ---------------------------------------------------------------------------
// Stub global fetch so useEffect fetch calls in page.tsx don't throw
// ---------------------------------------------------------------------------

global.fetch = jest.fn(() =>
  Promise.resolve({ json: () => Promise.resolve({ available: ["cpu"], default: "cpu" }) })
) as jest.Mock;

// ---------------------------------------------------------------------------
// Mock the WebSocket hook so tests never open a real connection
// ---------------------------------------------------------------------------

jest.mock("@/hooks/useVisualizerSocket", () => ({
  useVisualizerSocket: jest.fn(() => ({
    state: {
      connected: false,
      loading: false,
      frame: null,
      attention: null,
      norms: null,
      metrics: null,
      token_layout: null,
      config: null,
      events: [],
      reconstruction: null,
      error_map: null,
      reconstruction_error: null,
    },
    sendControl: jest.fn(),
  })),
}));

// ---------------------------------------------------------------------------
// Mock next/link so it renders a plain <a> (jsdom has no Next.js router)
// ---------------------------------------------------------------------------

jest.mock("next/link", () => {
  const MockLink = ({
    href,
    children,
    ...rest
  }: {
    href: string;
    children: React.ReactNode;
    [k: string]: unknown;
  }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  );
  MockLink.displayName = "Link";
  return MockLink;
});

// ---------------------------------------------------------------------------
// Import pages under test *after* mocks are set up
// ---------------------------------------------------------------------------

import LatentPage from "@/app/latent/page";
import HomePage from "@/app/page";

const mockUseVisualizerSocket = useVisualizerSocket as jest.Mock;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("LatentPage", () => {
  beforeEach(() => {
    mockUseVisualizerSocket.mockClear();
  });

  it("connects to /ws/latent so the backend computes reconstruction", () => {
    render(<LatentPage />);
    expect(mockUseVisualizerSocket).toHaveBeenCalledWith(
      undefined,
      undefined,
      undefined,
      "/ws/latent",
    );
  });

  it("renders three panes: raw frame, reconstruction, and error map", () => {
    const { container } = render(<LatentPage />);

    expect(container.querySelector("[data-testid='pane-raw']")).toBeInTheDocument();
    expect(container.querySelector("[data-testid='pane-reconstruction']")).toBeInTheDocument();
    expect(container.querySelector("[data-testid='pane-error-map']")).toBeInTheDocument();
  });

  it("shows a placeholder (not an error) when reconstruction is null", () => {
    render(<LatentPage />);
    // Placeholder text appears in both the reconstruction pane and the error map pane
    const placeholders = screen.getAllByText(/waiting for data/i);
    expect(placeholders.length).toBeGreaterThanOrEqual(1);
    // No JS error should be thrown — the test itself passing proves this
  });

  it("has a link back to the main page", () => {
    render(<LatentPage />);
    const backLink = screen.getByRole("link", { name: /main view/i });
    expect(backLink).toHaveAttribute("href", "/");
  });
});

describe("Navigation", () => {
  it("main page has a 'Latent View' link to /latent", () => {
    render(<HomePage />);
    const link = screen.getByRole("link", { name: /latent view/i });
    expect(link).toHaveAttribute("href", "/latent");
  });
});
